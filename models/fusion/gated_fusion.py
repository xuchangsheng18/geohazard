"""
Gated Multimodal Fusion (Mask-Aware Version)

Architecture rationale:
  Gated fusion mechanisms provide a lightweight yet effective way to combine
  features from different modalities. The gate network learns to weight each
  modality dynamically per sample.

  🔥 本地质灾害版本核心升级 (Mask-Aware):
  针对传感器 (Sensor) 经常缺失、文本报告 (Text) 有时为空的实际工程情况，
  引入了动态掩码机制 (Dynamic Masking)。缺失的模态在进入 Softmax 之前
  会被打上 -1e9 的负无穷惩罚，确保绝对不会稀释其他健康模态的注意力权重。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger


class GatedFusion(nn.Module):
    """Gated multimodal fusion: learns per-modality weights dynamically."""

    def __init__(self, feature_dim: int, num_modalities: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities

        logger.info(f"Initializing Mask-Aware GatedFusion (Modalities: {num_modalities})")

        # Gate network: concat all modalities → MLP → Logits (注意：这里不带 Softmax)
        gate_input_dim = feature_dim * num_modalities
        self.gate_logits_layer = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_modalities),
            # 🔥 移除了 Softmax，放到 forward 里手动计算，以便做 Mask 处理
        )

        # Post-fusion refinement
        self.refine = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        vision_feat: torch.Tensor,
        text_feat: torch.Tensor | None = None,
        sensor_feat: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            vision_feat: [B, dim] 视觉特征 (永远存在)
            text_feat:   [B, dim] 文本特征 (可能缺失/None)
            sensor_feat: [B, dim] 传感器特征 (极易缺失/None)

        Returns:
            fused: [B, feature_dim]
        """
        B, dim = vision_feat.shape
        device = vision_feat.device

        # 1. 动态对齐与构建掩码 (Mask)
        modalities = []
        valid_mask = []

        # 视觉 (Vision) 永远存在
        modalities.append(vision_feat)
        valid_mask.append(1.0)

        # 文本 (Text) 补全
        if text_feat is not None:
            modalities.append(text_feat)
            valid_mask.append(1.0)
        else:
            modalities.append(torch.zeros(B, dim, device=device))
            valid_mask.append(0.0)

        # 传感器 (Sensor) 补全
        if sensor_feat is not None:
            modalities.append(sensor_feat)
            valid_mask.append(1.0)
        else:
            modalities.append(torch.zeros(B, dim, device=device))
            valid_mask.append(0.0)

        # 转换为 Tensor: mask shape -> [B, num_modalities]
        valid_mask = torch.tensor(valid_mask, device=device).unsqueeze(0).expand(B, -1)

        # 2. 计算门控权重 (Gating)
        stacked = torch.stack(modalities, dim=1)        # [B, K, dim]
        gate_input = torch.cat(modalities, dim=-1)      # [B, dim * K]

        # 得到原始打分 (Logits)
        gate_logits = self.gate_logits_layer(gate_input) # [B, K]

        # 🔥 核心防御机制：把缺失模态的得分变为 -1e9 (负无穷)
        # 这样在 Softmax 之后，它们的权重必定为 0，且不占用总和的 1.0
        gate_logits = gate_logits.masked_fill(valid_mask == 0, -1e9)

        # 安全计算概率
        gate_weights = torch.softmax(gate_logits, dim=-1) # [B, K]

        # 3. 加权求和 (Weighted sum)
        # [B, K, dim] * [B, K, 1] -> sum over K -> [B, dim]
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)

        return self.refine(fused)