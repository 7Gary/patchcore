"""Pseudo anomaly synthesis inspired by recent industrial AD research."""

from __future__ import annotations

import random
from typing import Tuple

import torch


class CutPasteGenerator:
    """Implements CutPaste augmentation from Li et al., CVPR 2021."""

    def __init__(
        self,
        area_ratio_range: Tuple[float, float] = (0.02, 0.15),
        aspect_ratio_range: Tuple[float, float] = (0.3, 3.3),
        color_jitter: float = 0.05,
    ) -> None:
        self.area_ratio_range = area_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.color_jitter = color_jitter

    def __call__(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a pseudo anomalous variant of ``image`` and its mask."""

        if image.ndim != 3:
            raise ValueError("CutPasteGenerator expects a CHW tensor.")

        c, h, w = image.shape
        device = image.device

        area = h * w
        target_area = random.uniform(*self.area_ratio_range) * area
        aspect_ratio = random.uniform(*self.aspect_ratio_range)

        patch_h = max(1, int(round((target_area * aspect_ratio) ** 0.5)))
        patch_w = max(1, int(round((target_area / aspect_ratio) ** 0.5)))
        patch_h = min(patch_h, h)
        patch_w = min(patch_w, w)

        top = random.randint(0, h - patch_h)
        left = random.randint(0, w - patch_w)

        patch = image[:, top : top + patch_h, left : left + patch_w].clone()

        if self.color_jitter > 0:
            noise = torch.randn_like(patch) * self.color_jitter
            patch = torch.clamp(patch + noise, -1.0, 1.0)

        paste_top = random.randint(0, h - patch_h)
        paste_left = random.randint(0, w - patch_w)

        augmented = image.clone()
        augmented[:, paste_top : paste_top + patch_h, paste_left : paste_left + patch_w] = patch

        mask = torch.zeros((1, h, w), dtype=torch.float32, device=device)
        mask[:, paste_top : paste_top + patch_h, paste_left : paste_left + patch_w] = 1.0

        return augmented, mask
