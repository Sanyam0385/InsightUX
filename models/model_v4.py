"""
InsightUX GazeCNN v4 — EfficientNet-B0 backbone
================================================
WHY EfficientNet-B0 over ResNet-18:
  - ResNet-18 : 11.7M params, designed for 224x224 images
  - EfficientNet-B0: 5.3M params, designed for 224x224 but scales
    much better to small inputs like our 36x60 eye patches.
  - EfficientNet uses depthwise separable convolutions which are
    better at capturing fine-grained texture (iris patterns) with
    fewer parameters — less overfitting risk.
  - MobileNetV3 was also considered but EfficientNet-B0 consistently
    outperforms it on small image regression tasks.
  - ~4.5 deg expected val error (vs 5.89 in v2, ~8.3 stuck in v3).

Architecture:
  Stream A (binocular eye patches):
      Input : (B, 2, 36, 60)  <- left & right stacked as 2-channel
      EfficientNet-B0 (pretrained, first conv replaced for 2ch)
      AdaptiveAvgPool -> (B, 1280)
      FC(1280->256), BN, ReLU, Dropout(0.4) -> (B, 256)

  Stream B (head pose):
      Input : (B, 3)
      Linear(3->64), BN, ReLU, Dropout(0.1) -> (B, 64)

  Fusion:
      Concat -> (B, 320)
      BN(320)
      FC(320->128), ReLU, Dropout(0.3)
      FC(128->64),  ReLU, Dropout(0.2)
      FC(64->2) -> [pitch_rad, yaw_rad]
"""

import math
import torch
import torch.nn as nn
import torchvision.models as tv_models


class GazeCNNv4(nn.Module):

    def __init__(self):
        super().__init__()

        # ── Stream A: EfficientNet-B0 backbone ──────────────────────────────
        eff = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Replace first conv: original is (32, 3, 3, 3), we need 2 input channels
        orig_w = eff.features[0][0].weight.data   # (32, 3, 3, 3)
        new_w  = orig_w[:, :2, :, :].clone()      # keep first 2 channels
        new_w  = new_w * (3.0 / 2.0)              # rescale for same activation magnitude

        eff.features[0][0] = nn.Conv2d(
            2, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        eff.features[0][0].weight.data = new_w

        # Remove classifier, keep feature extractor + adaptive pool
        self.backbone = eff.features          # outputs (B, 1280, H', W')
        self.pool     = nn.AdaptiveAvgPool2d(1)

        self.fc_a = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
        )   # -> (B, 256)

        # ── Stream B: head pose MLP (expanded) ──────────────────────────────
        self.stream_b = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )   # -> (B, 64)

        # ── Fusion head ──────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(256 + 64),       # 320
            nn.Linear(320, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(64, 2),
        )   # -> (B, 2)  [pitch_rad, yaw_rad]

        self._init_new_layers()

    def _init_new_layers(self):
        """Initialise only the layers we added (backbone weights kept from ImageNet)."""
        for m in [self.fc_a, self.stream_b, self.fusion]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, patch: torch.Tensor,
                head_pose: torch.Tensor) -> torch.Tensor:
        # Stream A
        feat_a = self.pool(self.backbone(patch)).flatten(1)   # (B, 1280)
        feat_a = self.fc_a(feat_a)                             # (B, 256)

        # Stream B
        feat_b = self.stream_b(head_pose)                      # (B, 64)

        # Fusion
        fused = torch.cat([feat_a, feat_b], dim=1)             # (B, 320)
        return self.fusion(fused)                               # (B, 2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone_early(self):
        """Phase 1: freeze only the first 4 blocks (generic edge detectors)."""
        # EfficientNet features[0..3] = stem + MBConv blocks 1-3 (low-level)
        # features[4..8] = MBConv blocks 4-7 (high-level, need fine-tuning)
        for i, block in enumerate(self.backbone):
            freeze = i < 4
            for p in block.parameters():
                p.requires_grad = not freeze

    def unfreeze_all(self):
        """Phase 2: unfreeze everything."""
        for p in self.parameters():
            p.requires_grad = True


def angular_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Geodesic (cosine) loss — directly matches the angular error metric.
    Range [0, 2]. Lower is better.
    """
    def to_unit_vec(a):
        pitch, yaw = a[:, 0], a[:, 1]
        x = torch.cos(pitch) * torch.sin(yaw)
        y = torch.sin(pitch)
        z = torch.cos(pitch) * torch.cos(yaw)
        v = torch.stack([x, y, z], dim=1)
        return v / (v.norm(dim=1, keepdim=True) + 1e-8)

    cos_sim = (to_unit_vec(pred) * to_unit_vec(target)).sum(dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
    return (1.0 - cos_sim).mean()


def angular_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Evaluation metric: mean angular error in degrees."""
    def to_unit_vec(a):
        pitch, yaw = a[:, 0], a[:, 1]
        x = torch.cos(pitch) * torch.sin(yaw)
        y = torch.sin(pitch)
        z = torch.cos(pitch) * torch.cos(yaw)
        v = torch.stack([x, y, z], dim=1)
        return v / (v.norm(dim=1, keepdim=True) + 1e-8)

    cos_sim = (to_unit_vec(pred) * to_unit_vec(target)).sum(dim=1).clamp(-1, 1)
    return torch.acos(cos_sim) * (180.0 / math.pi)


if __name__ == "__main__":
    model = GazeCNNv4()
    print(f"Total parameters: {model.count_parameters():,}")

    model.freeze_backbone_early()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable (Phase 1, partial freeze): {trainable:,}")

    model.unfreeze_all()
    print(f"Trainable (Phase 2, all): {model.count_parameters():,}")

    patch = torch.randn(4, 2, 36, 60)
    pose  = torch.randn(4, 3)
    out   = model(patch, pose)
    print(f"Output shape: {out.shape}")   # (4, 2)
