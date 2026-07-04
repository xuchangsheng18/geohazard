"""
Transformer encoder for multivariate time-series.

Implements PatchTST-style patching:
  1. Divide each channel's time-series into overlapping/non-overlapping patches.
  2. Project each patch into a token embedding.
  3. Add learnable position encoding.
  4. Pass through a standard Transformer Encoder (Vaswani et al., 2017).
  5. Aggregate tokens via mean pooling.

This design is robust to varying sequence lengths and captures both
short-range patterns (within a patch) and long-range dependencies
(across patches via self-attention).

Reference:
  Nie, Y., et al. "A Time Series is Worth 64 Words: Long-term
  Forecasting with Transformers." ICLR 2023.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Splits [B, C, T] into non-overlapping patches and embeds each."""

    def __init__(self, in_channels: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(in_channels * patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]

        Returns:
            tokens: [B, num_patches, d_model]
        """
        B, C, T = x.shape
        # Unfold time dim into patches
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # patches: [B, C, num_patches, patch_len]
        patches = patches.permute(0, 2, 1, 3).contiguous()   # [B, num_patches, C, patch_len]
        patches = patches.view(B, patches.size(1), -1)        # [B, num_patches, C * patch_len]
        return self.proj(patches)                             # [B, num_patches, d_model]


class TimeSeriesTransformer(nn.Module):
    """Transformer encoder for sensor time-series with patch embedding."""

    def __init__(
        self,
        in_channels: int = 6,
        seq_length: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        patch_len: int = 16,
        patch_stride: int = 8,
    ):
        super().__init__()
        self.d_model = d_model

        # Patchify input
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            patch_len=patch_len,
            stride=patch_stride,
            d_model=d_model,
        )

        # Number of patches
        num_patches = (seq_length - patch_len) // patch_stride + 1
        self.num_patches = num_patches

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Class token (optional, for global aggregation)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # Pre-LN (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] — raw sensor time-series

        Returns:
            embedding: [B, d_model] — global sequence representation
        """
        B = x.shape[0]

        # Patch + project
        tokens = self.patch_embed(x)                  # [B, N_patches, d_model]

        # Add position encoding
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, 1+N_patches, d_model]

        tokens = self.dropout(tokens)

        # Self-attention
        tokens = self.transformer(tokens)              # [B, 1+N_patches, d_model]
        tokens = self.norm(tokens)

        # Return CLS token as global embedding
        return tokens[:, 0, :]                         # [B, d_model]
