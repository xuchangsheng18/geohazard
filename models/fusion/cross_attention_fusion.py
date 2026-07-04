"""
Cross-Attention Multimodal Fusion

Architecture rationale (academic references):
  Cross-attention fusion is the dominant paradigm in modern multimodal learning,
  popularized by:

  - ViLT (Kim et al., ICML 2021): Vision-Language Transformer using cross-modal
    attention, where one modality's embeddings attend to another modality's output.
  - METER (Dou et al., CVPR 2022): Demonstrates that cross-attention fusion
    consistently outperforms concatenation and single-stream approaches.
  - ALBEF (Li et al., NeurIPS 2021): Align-before-fuse strategy.

  🔥 在当前地质灾害多模态纠偏架构中 (Vision-Language Fusion):
    - Query (Q) 来自 Visual Modality (YOLO特征)。它主导了空间检测和基础视觉感知。
    - Key (K) / Value (V) 来自 Text Modality (BERT提取的现场勘查报告语义)。
    - 运行机制：视觉网络在看到类似圆形的物体时，通过 Cross-Attention 向文本网络“提问”（“报告里写的是工人还是落石？”）。
      文本网络通过 Key/Value 返回语义上下文，指导视觉特征进行修正，从而完美压制“人头当落石”的误报。

  This is a **Late Cross-Attention Fusion** strategy.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """A single cross-attention layer: Q attends to K/V, with residual + FFN."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,          # [B, N_q, dim] (当前为 Vision)
        key_value: torch.Tensor,      # [B, N_kv, dim] (当前为 Text)
    ) -> torch.Tensor:
        # Cross-attention: query 看着 key_value 寻找线索
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        query = self.norm1(query + attn_out)

        # FFN
        ffn_out = self.ffn(query)
        query = self.norm2(query + ffn_out)
        return query


class CrossAttentionFusion(nn.Module):
    """Multi-layer cross-attention fusion module.

    让主导模态 (Vision) 在多层网络中反复向上下文模态 (Text) 提取信息，
    最终输出融合后的抗干扰高维特征。
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(
        self,
        query_modality: torch.Tensor,      # [B, dim] -> 比如 Vision 提取出的全局特征
        key_value_modality: torch.Tensor,  # [B, dim] 或 [B, N, dim] -> 比如 Text 提取出的语义特征
    ) -> torch.Tensor:
        """
        Args:
            query_modality:   主导提问的模态特征 (Vision)
            key_value_modality: 提供上下文答疑的模态特征 (Text)

        Returns:
            fused: [B, dim] 深度融合后的特征，用于最终的分类判断
        """
        # 兼容性设计：如果输入是纯向量 [B, dim]，扩展一个序列维度变为 [B, 1, dim]
        # 这确保了无论上游是输出单个 [CLS] token 还是多个区域 token，都能无缝对接
        if query_modality.ndim == 2:
            query_modality = query_modality.unsqueeze(1)
        if key_value_modality.ndim == 2:
            key_value_modality = key_value_modality.unsqueeze(1)

        q = query_modality
        for block in self.blocks:
            q = block(q, key_value_modality)

        # 压缩掉序列维度，返回 [B, dim] 交给最后的 Classifier
        fused = self.final_norm(q.squeeze(1))
        return fused