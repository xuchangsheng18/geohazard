"""
Time-series encoder for geological sensor data.

Design rationale (academic reference):
  We adopt a **PatchTST** + **Transformer Encoder** architecture, which has
  demonstrated state-of-the-art performance on long-sequence time-series tasks
  (Nie et al., ICLR 2023). Each input channel is independently patchified,
  then cross-channel information is fused via a shared Transformer encoder.

  For shorter sequences, a 1D-CNN (InceptionTime-style) is offered as an
  alternative backbone with far fewer parameters.

Reference:
  - PatchTST: "A Time Series is Worth 64 Words" (Nie et al., ICLR 2023)
  - Informer: "Informer: Beyond Efficient Transformer..." (Zhou et al., AAAI 2021)
  - TimesNet: "TimesNet: Temporal 2D-Variation Modeling..." (Wu et al., ICLR 2023)
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from loguru import logger

from models.timeseries.ts_transformer import TimeSeriesTransformer


class SensorEncoder1DCNN(nn.Module):
    """Lightweight 1D-CNN backbone — InceptionTime-style.

    Good for: limited training data, low-latency inference.
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_dim: int = 128,
        out_dim: int = 256,
        seq_length: int = 256,
    ):
        super().__init__()

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels if i == 0 else hidden_dim, hidden_dim,
                          kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            )
            for i, k in enumerate([7, 5, 3])  # multi-scale kernels
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] → embedding: [B, out_dim]"""
        for conv in self.conv_blocks:
            x = conv(x)
        x = self.pool(x).squeeze(-1)   # [B, hidden_dim]
        return self.fc(x)               # [B, out_dim]


class SensorEncoder(nn.Module):
    """Top-level sensor encoder — picks backbone automatically.

    For long sequences (≥ 96 steps), defaults to Transformer.
    For shorter sequences, defaults to 1D-CNN.
    """

    def __init__(
        self,
        in_channels: int = 6,
        seq_length: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        out_dim: int = 256,
        backbone: Literal["auto", "transformer", "cnn"] = "auto",
    ):
        super().__init__()
        self.seq_length = seq_length
        self.out_dim = out_dim

        if backbone == "auto":
            backbone = "transformer" if seq_length >= 96 else "cnn"

        logger.info(f"SensorEncoder using backbone: {backbone}")

        if backbone == "transformer":
            self.encoder = TimeSeriesTransformer(
                in_channels=in_channels,
                seq_length=seq_length,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
            )
            self.proj = nn.Linear(d_model, out_dim)
        else:
            self.encoder = SensorEncoder1DCNN(
                in_channels=in_channels,
                hidden_dim=128,
                out_dim=out_dim,
                seq_length=seq_length,
            )
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T] → embedding: [B, out_dim]"""
        h = self.encoder(x)      # [B, d_model] or [B, out_dim]
        return self.proj(h)      # [B, out_dim]
