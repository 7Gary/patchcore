"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device
        self.contamination_ratio = 0.0
        self.contamination_topk_ratio = 0.1
        self.target_false_positive = None
        self.target_miss_rate = None
        self.pseudo_anomaly_generator = None
        self.pseudo_calibration_max_batches = 0
        self.pseudo_samples_per_image = 1
        self.pseudo_lower_percentile = 0.1
        self.calibrated_threshold = None
        self.last_filtered_indices = []
        self.clean_score_distribution = []
        self.pseudo_score_distribution = []
        self.training_clean_threshold = None

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        contamination_ratio=0.0,
        contamination_topk_ratio=0.1,
        target_false_positive=0.05,
        target_miss_rate=0.05,
        pseudo_anomaly_generator=None,
        pseudo_calibration_max_batches=4,
        pseudo_samples_per_image=1,
        pseudo_lower_percentile=0.1,
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler
        self.contamination_ratio = contamination_ratio
        self.contamination_topk_ratio = contamination_topk_ratio
        self.target_false_positive = target_false_positive
        self.target_miss_rate = target_miss_rate
        self.pseudo_anomaly_generator = pseudo_anomaly_generator
        self.pseudo_calibration_max_batches = pseudo_calibration_max_batches
        self.pseudo_samples_per_image = pseudo_samples_per_image
        self.pseudo_lower_percentile = pseudo_lower_percentile
        self.calibrated_threshold = None

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for SPADE with noise filtering."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                features, patch_shapes = self._embed(
                    input_image, provide_patch_shapes=True
                )
            features = np.asarray(features)
            batchsize = input_image.shape[0]
            num_patches = patch_shapes[0][0] * patch_shapes[0][1]
            return features.reshape(batchsize, num_patches, -1)

        image_features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for batch in data_iterator:
                if isinstance(batch, dict):
                    batch = batch["image"]
                batch_features = _image_to_features(batch)
                image_features.extend([np.copy(feat) for feat in batch_features])

        (
            filtered_features,
            filtered_indices,
            per_image_scores,
            contamination_threshold,
        ) = self._filter_contaminated_embeddings(image_features)

        self.last_filtered_indices = filtered_indices
        self.clean_score_distribution = per_image_scores
        self.training_clean_threshold = contamination_threshold

        features = np.concatenate(filtered_features, axis=0)
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])

        if (
            self.pseudo_anomaly_generator is not None
            and self.pseudo_calibration_max_batches > 0
        ):
            self._calibrate_with_pseudo_anomalies(input_data)

    def _filter_contaminated_embeddings(self, image_features):
        """Filter noisy supports similar to Pseudo Multi-View Dual-Teacher (CVPR'23)."""
        if not image_features:
            return image_features, [], [], None

        if self.contamination_ratio <= 0:
            return image_features, [], [0.0 for _ in image_features], None

        stacked = np.concatenate(image_features, axis=0)
        feature_mean = np.mean(stacked, axis=0, keepdims=True)
        patch_scores = np.linalg.norm(stacked - feature_mean, axis=1)

        per_image_scores = []
        filtered_features = []
        filtered_indices = []

        start_idx = 0
        for idx, feats in enumerate(image_features):
            end_idx = start_idx + len(feats)
            image_patch_scores = patch_scores[start_idx:end_idx]
            start_idx = end_idx
            topk = max(1, int(np.ceil(len(image_patch_scores) * self.contamination_topk_ratio)))
            top_scores = np.partition(image_patch_scores, -topk)[-topk:]
            per_image_scores.append(float(np.mean(top_scores)))

        contamination_threshold = float(
            np.quantile(per_image_scores, 1 - self.contamination_ratio)
        )

        for idx, (feats, score) in enumerate(zip(image_features, per_image_scores)):
            if score <= contamination_threshold or len(filtered_features) == 0:
                filtered_features.append(feats)
            else:
                filtered_indices.append(idx)

        if not filtered_features:
            filtered_features = image_features
            filtered_indices = []

        if filtered_indices:
            LOGGER.info(
                "Filtered %d/%d suspected anomalous training tiles (threshold=%.4f).",
                len(filtered_indices),
                len(image_features),
                contamination_threshold,
            )

        return filtered_features, filtered_indices, per_image_scores, contamination_threshold

    def _calibrate_with_pseudo_anomalies(self, dataloader):
        """Calibrate thresholds using CutPaste pseudo anomalies (CVPR'21)."""
        clean_scores = []
        pseudo_scores = []
        batches_processed = 0

        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch["image"]
            else:
                images = batch

            with torch.no_grad():
                clean_batch_scores, _ = self._predict(images)

            pseudo_images = self._generate_pseudo_batch(images)
            with torch.no_grad():
                pseudo_batch_scores, _ = self._predict(pseudo_images)

            clean_scores.extend(clean_batch_scores)
            pseudo_scores.extend(pseudo_batch_scores)

            batches_processed += 1
            if batches_processed >= self.pseudo_calibration_max_batches:
                break

        if not clean_scores or not pseudo_scores:
            return

        target_miss = self.target_miss_rate if self.target_miss_rate is not None else 0.05
        upper_clean = float(np.percentile(clean_scores, (1 - target_miss) * 100))
        lower_pseudo = float(
            np.percentile(pseudo_scores, self.pseudo_lower_percentile * 100)
        )

        self.calibrated_threshold = max(upper_clean, (upper_clean + lower_pseudo) / 2.0)
        self.clean_score_distribution = clean_scores
        self.pseudo_score_distribution = pseudo_scores

        LOGGER.info(
            "Calibrated decision threshold at %.4f using %d pseudo batches.",
            self.calibrated_threshold,
            batches_processed,
        )

    def _generate_pseudo_batch(self, images):
        pseudo_images = []
        for image in images:
            pseudo_image = image.detach().cpu()
            for _ in range(max(1, self.pseudo_samples_per_image)):
                pseudo_image, _ = self.pseudo_anomaly_generator(pseudo_image)
            pseudo_images.append(pseudo_image)

        pseudo_batch = torch.stack(pseudo_images, dim=0).to(images.device)
        return pseudo_batch.to(dtype=images.dtype)

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
