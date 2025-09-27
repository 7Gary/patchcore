"""Custom dataset utilities tailored to textile inspection."""

import os
from enum import Enum
from typing import List, Optional, Tuple

import PIL
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class CustomDataset(torch.utils.data.Dataset):
    """Dataset abstraction for custom industrial inspection datasets.

    The dataset assumes the following directory structure::

        root/class_name/{train,val,test}/anomaly_type/*.png

    Each training image is split into two tiles (left/right) which mirrors
    the structure of the provided baseline code. During testing the same split
    is used and optional ground-truth masks can be provided via the fourth
    element of ``data_to_iterate``.
    """

    def __init__(
        self,
        source: str,
        classname: Optional[str],
        resize: int = 256,
        imagesize: int = 224,
        split: DatasetSplit = DatasetSplit.TRAIN,
        train_val_split: float = 1.0,
        augment: bool = False,
        classnames: Optional[List[str]] = None,
        **_,
    ) -> None:
        super().__init__()

        self.source = source
        self.split = split
        self.train_val_split = train_val_split
        if classname is not None:
            self.classnames_to_use = [classname]
        elif classnames is not None:
            self.classnames_to_use = classnames
        else:
            # Fallback for legacy behaviour when no explicit list is provided.
            self.classnames_to_use = sorted(
                [
                    d
                    for d in os.listdir(source)
                    if os.path.isdir(os.path.join(source, d))
                ]
            )

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.transform_mask = transforms.Compose(
            [
                transforms.Resize(resize, interpolation=PIL.Image.NEAREST),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
            ]
        )

        self.imagesize = (3, imagesize, imagesize)
        self.augment = augment and split == DatasetSplit.TRAIN

        if self.augment:
            # Mild appearance jittering to reduce overfitting to illumination.
            self.augment_img = transforms.Compose(
                [
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
                    ),
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))],
                        p=0.2,
                    ),
                ]
            )
        else:
            self.augment_img = None

    def __getitem__(self, idx: int):
        orig_idx = idx // 2
        slice_idx = idx % 2

        classname, anomaly, image_path, mask_path = self.data_to_iterate[orig_idx]
        full_image = PIL.Image.open(image_path).convert("RGB")

        tile = self._crop_tile(full_image, slice_idx)
        if self.augment_img is not None:
            tile = self.augment_img(tile)

        image = self.transform_img(tile)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            full_mask = PIL.Image.open(mask_path)
            mask_tile = self._crop_tile(full_mask, slice_idx)
            mask = self.transform_mask(mask_tile)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        slice_suffix = "_left" if slice_idx == 0 else "_right"
        image_name = "/".join(image_path.split("/")[-4:]) + slice_suffix

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": image_name,
            "image_path": image_path,
        }

    def __len__(self) -> int:
        return len(self.data_to_iterate) * 2

    @staticmethod
    def _crop_tile(image: PIL.Image.Image, slice_idx: int) -> PIL.Image.Image:
        width, height = image.size
        midpoint = width // 2
        if slice_idx == 0:
            bbox = (0, 0, midpoint, height)
        else:
            bbox = (midpoint, 0, width, height)
        return image.crop(bbox)

    def get_image_data(self) -> Tuple[dict, List[List[Optional[str]]]]:
        imgpaths_per_class = {}
        data_to_iterate: List[List[Optional[str]]] = []

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            if not os.path.exists(classpath):
                continue

            anomaly_types = sorted(
                [
                    anomaly
                    for anomaly in os.listdir(classpath)
                    if os.path.isdir(os.path.join(classpath, anomaly))
                ]
            )
            imgpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(
                    [
                        os.path.join(anomaly_path, f)
                        for f in os.listdir(anomaly_path)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    ]
                )

                if not anomaly_files:
                    continue

                if self.train_val_split < 1.0:
                    split_idx = int(len(anomaly_files) * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        anomaly_files = anomaly_files[:split_idx]
                    elif self.split == DatasetSplit.VAL:
                        anomaly_files = anomaly_files[split_idx:]

                imgpaths_per_class[classname][anomaly] = anomaly_files

                for image_path in anomaly_files:
                    data_to_iterate.append([classname, anomaly, image_path, None])

        return imgpaths_per_class, data_to_iterate
