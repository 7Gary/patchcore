from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureFuser(nn.Module):
    """Multi-scale feature融合模块，借鉴 CVPR 2023 RD++ 与 CVPR 2024 VisionAD 中的跨层注意思想."""

    def __init__(
        self,
        in_channels: Iterable[int],
        fusion_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_channels = list(in_channels)
        if len(in_channels) == 0:
            raise ValueError("in_channels 不能为空")

        self.num_scales = len(in_channels)
        self.fused_dim = fusion_dim
        self.projectors = nn.ModuleList()
        self.scale_mlps = nn.ModuleList()

        hidden_dim = max(self.fused_dim // 2, 1)
        for channel in in_channels:
            self.projectors.append(
                nn.Sequential(
                    nn.Conv2d(channel, self.fused_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.fused_dim),
                    nn.GELU(),
                )
            )
            self.scale_mlps.append(
                nn.Sequential(
                    nn.Linear(self.fused_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, 1),
                )
            )

        attn_heads = max(1, min(num_heads, self.fused_dim // 16))
        self.attention = nn.MultiheadAttention(
            self.fused_dim, attn_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(self.fused_dim)
        self.norm2 = nn.LayerNorm(self.fused_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.fused_dim, self.fused_dim * 4),
            nn.GELU(),
            nn.Linear(self.fused_dim * 4, self.fused_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.spatial_refiner = nn.Sequential(
            nn.Conv2d(self.fused_dim, self.fused_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fused_dim),
            nn.GELU(),
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        if not isinstance(features, (list, tuple)):
            raise TypeError("features 必须为张量列表")
        if len(features) != self.num_scales:
            raise ValueError(
                f"期望 {self.num_scales} 个尺度特征，实际得到 {len(features)}"
            )

        target_h = min(feature.shape[-2] for feature in features)
        target_w = min(feature.shape[-1] for feature in features)
        base_h = max(feature.shape[-2] for feature in features)
        base_w = max(feature.shape[-1] for feature in features)

        projected_features = []
        scale_logits = []
        for feature, projector, mlp in zip(
            features, self.projectors, self.scale_mlps
        ):
            aligned = F.adaptive_avg_pool2d(feature, output_size=(target_h, target_w))
            projected = projector(aligned)
            projected_features.append(projected)
            pooled = torch.flatten(F.adaptive_avg_pool2d(projected, 1), 1)
            scale_logits.append(mlp(pooled))

        scale_logits = torch.cat(scale_logits, dim=-1)
        scale_weights = torch.softmax(scale_logits, dim=-1)

        modulated_features: List[torch.Tensor] = []
        for feature, weight in zip(features, torch.unbind(scale_weights, dim=-1)):
            weight = weight.view(weight.size(0), 1, 1, 1)
            modulated_features.append(feature * (1.0 + weight))

        fused = 0.0
        for weight, projected in zip(
            torch.unbind(scale_weights, dim=-1), projected_features
        ):
            fused = fused + projected * weight.view(weight.size(0), 1, 1, 1)

        fused = self.spatial_refiner(fused)
        b, c, h, w = fused.shape
        tokens = fused.view(b, c, h * w).transpose(1, 2)
        attn_output, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + self.dropout(attn_output))
        ffn_output = self.ffn(tokens)
        tokens = self.norm2(tokens + self.dropout(ffn_output))
        fused = tokens.transpose(1, 2).view(b, c, h, w)

        if (h, w) != (base_h, base_w):
            fused = F.interpolate(
                fused, size=(base_h, base_w), mode="bilinear", align_corners=False
            )

        return modulated_features + [fused]