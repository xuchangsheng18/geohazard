"""
Transformer-Based Multimodal Fusion

Architecture rationale:
  The "multi-modal transformer" approach treats each modality's feature as
  a set of tokens, then feeds them jointly into a Transformer encoder.
  Self-attention across all modality tokens allows the model to learn
  arbitrary inter-modal interactions.

  This design follows the paradigm established by:
  - VisualBERT (Li et al., 2019): Single-stream Transformer where image
    region features and text tokens are concatenated as input.
  - UNITER (Chen et al., ECCV 2020): Unified Transformer with
    Image-Text Matching pre-training.
  - Perceiver IO (Jaegle et al., ICML 2022): Uses learned latent queries
    to cross-attend to a concatenated multi-modal byte array, decoupling
    input size from compute.

  In our system:
    - Each modality produces a fixed set of "modality tokens" (via projection).
    - A modality type embedding is added so the model knows each token's origin.
    - A standard TransformerEncoder processes all tokens jointly.
    - The output CLS token or mean-pooled embedding is the fused representation.

Reference:
  - Li, L.H., et al. "VisualBERT: A Simple and Performant Baseline for
    Vision and Language." 2019.
  - Chen, Y.-C., et al. "UNITER: UNiversal Image-TExt Representation
    Learning." ECCV 2020.
"""
from __future__ import annotations

import torch
import torch.nn as nn


from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger


class TransformerFusion(nn.Module):
    """Transformer-based multimodal fusion with dynamic modality support."""

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_modalities: int = 3,  # Vision, Text, Sensor
        tokens_per_modality: int = 4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities
        self.tokens_per_modality = tokens_per_modality

        # Project each modality's single vector into multiple tokens
        self.token_projectors = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim * tokens_per_modality)
            for _ in range(num_modalities)
        ])

        # Modality type embeddings
        self.modality_embed = nn.Embedding(num_modalities, feature_dim)

        # Position embeddings (allow for max tokens)
        total_tokens = num_modalities * tokens_per_modality
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, feature_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, *modalities: torch.Tensor | None) -> torch.Tensor:
        """
        Args:
            modalities: 可变参数，支持 None (如果某个模态缺失)
        """
        # 过滤掉 None，保留存在的模态
        valid_modalities = [(i, m) for i, m in enumerate(modalities) if m is not None]
        B = next(m for m in modalities if m is not None).shape[0]
        device = next(m for m in modalities if m is not None).device

        tokens_list = []
        for mod_idx, mod_feat in valid_modalities:
            # 1. 投影成多个 Token
            mod_tokens = self.token_projectors[mod_idx](mod_feat)  # [B, dim * T]
            mod_tokens = mod_tokens.view(B, self.tokens_per_modality, self.feature_dim)

            # 2. 加上模态类型 Embedding (mod_idx 为下标)
            type_emb = self.modality_embed(torch.tensor([mod_idx], device=device))
            mod_tokens = mod_tokens + type_emb.unsqueeze(1) # [B, T, dim]

            tokens_list.append(mod_tokens)

        # 3. 拼接所有 Token
        all_tokens = torch.cat(tokens_list, dim=1)   # [B, N_total, dim]

        # 4. 加入位置编码 (只取对应的长度)
        all_tokens = all_tokens + self.pos_embed[:, :all_tokens.shape[1], :]

        # 5. 加入 CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        all_tokens = torch.cat([cls_tokens, all_tokens], dim=1)

        # 6. Transformer 处理
        all_tokens = self.dropout(all_tokens)
        all_tokens = self.transformer(all_tokens)
        all_tokens = self.norm(all_tokens)

        return all_tokens[:, 0, :]   # [B, dim]
