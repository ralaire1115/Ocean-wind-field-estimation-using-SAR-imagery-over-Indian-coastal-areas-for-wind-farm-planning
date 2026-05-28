"""

Architecture
------------
A modified ResNet-18 (pretrained on ImageNet) is used as the backbone:

  Input  : (B, 1, H, W)  – single-channel normalised SAR σ₀ patch
  Stem   : Conv2d(1 → 64) replacing the original 3-channel first layer
  Trunk  : ResNet-18 feature layers (layer1 … layer4)
  Head   : AdaptiveAvgPool → Flatten → Linear(512 → 2)
  Output : (B, 2)  →  [wind_speed (m/s), wind_direction (°)]

Notes
-----
- The model is **untrained** in this skeleton.  Fine-tune it on a labelled
  dataset of (SAR patch → ERA5 / ECMWF wind speed + direction) pairs.
- Wind direction is constrained to [0, 360) via a sigmoid-scaled output.
- Wind speed is constrained to [0, 40] m/s (typical ocean wind range).
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

logger = logging.getLogger(__name__)

# Constants

WIND_SPEED_MAX   = 40.0    # m/s  – used to scale sigmoid output
WIND_DIR_MAX     = 360.0   # degrees
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Output dataclass

@dataclass
class WindVector:
    """Structured container for a single wind-field prediction."""
    wind_speed_ms:  float   # Estimated wind speed  (m/s)
    wind_dir_deg:   float   # Estimated wind direction (meteorological, 0-360°)


# Model definition

class SARWindNet(nn.Module):
    """
    ResNet-18 backbone adapted for single-channel SAR → wind regression.

    Modifications vs. stock ResNet-18
    ----------------------------------
    1. First Conv2d changed from in_channels=3 to in_channels=1 so the model
       accepts greyscale SAR patches without RGB duplication artefacts.
    2. Final fully-connected layer replaced with ``Linear(512 → 2)`` for
       two-value regression (wind speed, wind direction).
    3. A custom ``_constrain_output`` method enforces physically meaningful
       output ranges via sigmoid scaling.
    """

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()

        # Load ResNet-18 base
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Patch stem: 3-channel → 1-channel 
        original_conv = backbone.conv1           # Conv2d(3, 64, 7, stride=2, padding=3)
        backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        if pretrained:
            # Initialise the new 1-channel conv by averaging the 3 RGB channels
            with torch.no_grad():
                backbone.conv1.weight.copy_(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

        # Keep everything except the original classifier
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,           # AdaptiveAvgPool2d → (B, 512, 1, 1)
        )

        # Custom regression head 
        in_features: int = backbone.fc.in_features   # 512 for ResNet-18
        self.regressor = nn.Linear(in_features, 2)

        logger.info(
            "SARWindNet initialised | device=%s | pretrained=%s",
            DEVICE, pretrained,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, 1, H, W)``, values in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 2)`` → ``[:, 0]`` wind speed (m/s),
                               ``[:, 1]`` wind direction (°).
        """
        features = self.feature_extractor(x)          # (B, 512, 1, 1)
        features = torch.flatten(features, start_dim=1)  # (B, 512)
        raw      = self.regressor(features)            # (B, 2)  raw logits
        return self._constrain_output(raw)

    @staticmethod
    def _constrain_output(raw: torch.Tensor) -> torch.Tensor:
        """
        Map unbounded raw regression outputs to physically valid ranges:

        - Wind speed  : sigmoid(raw[:, 0]) × WIND_SPEED_MAX  → [0, 40] m/s
        - Wind dir    : sigmoid(raw[:, 1]) × WIND_DIR_MAX    → [0, 360]°
        """
        speed = torch.sigmoid(raw[:, 0]) * WIND_SPEED_MAX    # [0, 40]
        direc = torch.sigmoid(raw[:, 1]) * WIND_DIR_MAX      # [0, 360)
        return torch.stack([speed, direc], dim=1)


# Lazy model singleton (loaded once per process)

_model: SARWindNet | None = None


def _get_model() -> SARWindNet:
    """
    Return the singleton ``SARWindNet`` instance.

    In production, load pre-trained weights here:
    ``model.load_state_dict(torch.load('sar_wind_weights.pth'))``
    """
    global _model
    if _model is None:
        _model = SARWindNet(pretrained=False).to(DEVICE)
        _model.eval()
        logger.info("SARWindNet loaded onto device: %s", DEVICE)
    return _model


# Public API

def calculate_wind_vectors(sar_image_array: np.ndarray) -> WindVector:
    """
    Run inference on a single SAR image patch and return wind speed and
    direction estimates.

    Parameters
    ----------
    sar_image_array : np.ndarray
        Shape ``(1, H, W)`` or ``(H, W)``, dtype ``float32``, values in
        ``[0, 1]``.  Typically produced by ``gee_engine.fetch_sar_image()``.

    Returns
    -------
    WindVector
        ``.wind_speed_ms``  – estimated wind speed in m/s
        ``.wind_dir_deg``   – estimated meteorological wind direction in °

    Raises
    ------
    ValueError
        If the input array has an unexpected shape or dtype.
    """
    # 1. Validate and reshape input
    arr = np.asarray(sar_image_array, dtype=np.float32)

    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]        # (H, W) → (1, H, W)

    if arr.ndim != 3 or arr.shape[0] != 1:
        raise ValueError(
            f"Expected SAR array of shape (1, H, W) or (H, W), "
            f"got {arr.shape}."
        )

    # 2. Build a (1, 1, H, W) tensor  [batch=1, channel=1]
    tensor: torch.Tensor = (
        torch.from_numpy(arr)
        .unsqueeze(0)            # (1, 1, H, W)
        .to(DEVICE)
    )

    # 3. Inference (no gradient needed)
    model = _get_model()

    with torch.no_grad():
        predictions: torch.Tensor = model(tensor)    # (1, 2)

    speed_ms: float = predictions[0, 0].item()
    dir_deg:  float = predictions[0, 1].item()

    logger.info(
        "Wind inference complete | speed=%.2f m/s | direction=%.1f°",
        speed_ms, dir_deg,
    )

    return WindVector(wind_speed_ms=round(speed_ms, 3),
                      wind_dir_deg=round(dir_deg, 3))
