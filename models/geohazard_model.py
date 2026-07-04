"""
GeoHazard Multimodal Model — Top-Level Architecture

This is the MAIN model class that wires together all modality encoders
and the fusion module into a single end-to-end trainable/inferenceable model.

灾害类别 (4 类，与 YOLO data.yaml 一致):
    0: fallen tree      倒树
    1: landslide        滑坡
    2: road collapse    道路塌陷
    3: stone            落石

Full data flow:

    ┌─────────────────────────────────────────────────────────────────┐
    │                     GeoHazardMultimodalModel                     │
    │                                                                  │
    │  Image ──→ YOLOFeatureExtractor (best.pt) ──→ [B, D_vis]       │
    │                                                     ──→ Fusion  │
    │  Sensor ─→ SensorEncoder (PatchTST) ────────────→ [B, D_ts]     │
    │                                                     │           │
    │  Text ───→ TextEncoder (BERT) ──────────────────→ [B, D_txt]    │
    │                                                     │           │
    │  Fused ─→ Classification Head ─→ [B, 4]                        │
    └─────────────────────────────────────────────────────────────────┘

Architecture design choices (with academic references):

1. **Modality-specific encoders** — Each modality uses a dedicated encoder
   optimized for its data type:
   - Vision: YOLO11 (pre-trained on hazard images) — extracts spatial features
   - Time-series: PatchTST-based Transformer — captures long-range temporal
     dependencies in sensor data
   - Text: BERT/RoBERTa — encodes Chinese geological field reports

2. **Multimodal Fusion** — Three strategies implemented (config-switchable):
   - ``cross_attention``: Sensor-driven cross-attention over visual features
     (inspired by ViLT/METER). Recommended when sensor data is the primary
     decision signal.
   - ``transformer``: Single-stream joint Transformer (inspired by VisualBERT/
     UNITER). Recommended when all modalities are equally informative.
   - ``gated``: Dynamic per-sample modality weighting (inspired by Gated
     Multimodal Units). Recommended when modalities may be missing or noisy.

Reference compilation:
  - ViLT: Kim et al., ICML 2021
  - METER: Dou et al., CVPR 2022
  - PatchTST: Nie et al., ICLR 2023
  - BERT: Devlin et al., NAACL 2019
  - Gated Fusion: Arevalo et al., ICLR-W 2017
  - UNITER: Chen et al., ECCV 2020
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from loguru import logger

from configs.model_config import GeoHazardConfig
from models.vision.yolo_encoder import YOLOFeatureExtractor
from models.timeseries.sensor_encoder import SensorEncoder
from models.text.text_encoder import TextEncoder
from models.fusion.cross_attention_fusion import CrossAttentionFusion
from models.fusion.gated_fusion import GatedFusion
from models.fusion.transformer_fusion import TransformerFusion


class ClassificationHead(nn.Module):
    """Simple MLP classification head with LayerNorm + GELU + Dropout."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(x))


class GeoHazardMultimodalModel(nn.Module):
    """End-to-end multimodal model for geological hazard detection.

    4 类输出: fallen tree | landslide | road collapse | stone

    Usage::

        config = GeoHazardConfig(...)
        model = GeoHazardMultimodalModel(config)

        # Forward：图像/传感器/文本 → 灾害分类 logits
        logits, embeddings = model(images, sensors, reports)
    """

    def __init__(self, config: GeoHazardConfig):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.fusion_method = config.fusion.method

        # -----------------------------------------------------------------
        # 1. Visual encoder — 加载你训练好的 YOLO11 best.pt
        # -----------------------------------------------------------------
        logger.info("=" * 60)
        logger.info(f"Initializing Visual Encoder (YOLO11 + {config.vision.custom_weights})")
        logger.info(f"Detection classes: {config.class_names}")
        logger.info("=" * 60)
        self.vision_encoder = YOLOFeatureExtractor(
            weights_path=config.vision.custom_weights,
            feature_dim=config.vision.feature_dim,
            input_size=config.vision.input_size,
            freeze=config.vision.freeze_backbone,
        )

        # -----------------------------------------------------------------
        # 2. Time-series encoder (PatchTST Transformer)
        # -----------------------------------------------------------------
        if config.timeseries.enabled:
            logger.info("Initializing Time-Series Encoder (PatchTST)")
            self.sensor_encoder = SensorEncoder(
                in_channels=config.timeseries.input_channels,
                seq_length=config.timeseries.seq_length,
                d_model=config.timeseries.d_model,
                n_heads=config.timeseries.n_heads,
                n_layers=config.timeseries.n_layers,
                dropout=config.timeseries.dropout,
                out_dim=config.timeseries.feature_dim,
            )
        else:
            self.sensor_encoder = None

        # -----------------------------------------------------------------
        # 3. Text encoder (BERT / RoBERTa)
        # -----------------------------------------------------------------
        if config.text.enabled:
            logger.info("Initializing Text Encoder (BERT)")
            self.text_encoder = TextEncoder(
                model_name=config.text.model_name,
                max_length=config.text.max_length,
                feature_dim=config.text.feature_dim,
                freeze_backbone=config.text.freeze_backbone,
            )
        else:
            self.text_encoder = None

        # -----------------------------------------------------------------
        # 4. Fusion module
        # -----------------------------------------------------------------
        logger.info(f"Initializing Fusion: {config.fusion.method}")
        self.fusion = self._build_fusion(config)

        # 确定融合后特征维度
        if config.fusion.method == "gated":
            fused_dim = config.vision.feature_dim
        else:
            fused_dim = config.fusion.hidden_dim

        # -----------------------------------------------------------------
        # 5. Classification head → 4 类输出
        # -----------------------------------------------------------------
        self.classifier = ClassificationHead(
            in_dim=fused_dim,
            num_classes=config.num_classes,
            hidden_dim=fused_dim // 2,
        )

        logger.info(
            f"GeoHazardMultimodalModel initialized. "
            f"num_classes={config.num_classes}, fusion={config.fusion.method}"
        )

    def _build_fusion(self, config: GeoHazardConfig) -> nn.Module:
        method = config.fusion.method
        fc = config.fusion

        if method == "cross_attention":
            return CrossAttentionFusion(
                dim=fc.hidden_dim,
                num_heads=fc.num_heads,
                num_layers=fc.num_layers,
                dropout=fc.dropout,
            )
        elif method == "gated":
            n_mods = 1  # vision always active
            if config.timeseries.enabled:
                n_mods += 1
            if config.text.enabled:
                n_mods += 1
            return GatedFusion(
                feature_dim=config.vision.feature_dim,
                num_modalities=n_mods,
                hidden_dim=fc.hidden_dim // 2,
            )
        elif method == "transformer":
            n_mods = 1 + int(config.timeseries.enabled) + int(config.text.enabled)
            return TransformerFusion(
                feature_dim=fc.hidden_dim,
                num_heads=fc.num_heads,
                num_layers=fc.num_layers,
                dropout=fc.dropout,
                num_modalities=n_mods,
                tokens_per_modality=4,
            )
        elif method == "concat":
            total_dim = config.vision.feature_dim
            if config.timeseries.enabled:
                total_dim += config.timeseries.feature_dim
            if config.text.enabled:
                total_dim += config.text.feature_dim
            return nn.Sequential(
                nn.Linear(total_dim, fc.hidden_dim),
                nn.LayerNorm(fc.hidden_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def forward(
        self,
        images: torch.Tensor,
        sensors: Optional[torch.Tensor] = None,
        reports: Optional[list[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images:  [B, 3, H, W]    — RGB 图像
            sensors: [B, C, T]        — 传感器时序（可选）
            reports: list[str]        — 现场报告文本（可选）

        Returns:
            logits:     [B, 4]  — 4 类灾害 logits
            embeddings: [B, fused_dim] — 融合后 embedding
        """
        embeddings = []

        # --- Vision ---
        vis_emb = self.vision_encoder.forward_embedding(images, pool="attention")
        embeddings.append(vis_emb)

        # --- Time-Series ---
        if self.sensor_encoder is not None and sensors is not None:
            ts_emb = self.sensor_encoder(sensors)
            embeddings.append(ts_emb)

        # --- Text ---
        if self.text_encoder is not None and reports is not None and len(reports) > 0:
            txt_emb = self.text_encoder(reports)
            embeddings.append(txt_emb)

        # --- Fusion ---
        if self.fusion_method == "concat":
            fused = torch.cat(embeddings, dim=-1)
            fused = self.fusion(fused)
        elif self.fusion_method == "cross_attention":
            q = ts_emb if self.sensor_encoder is not None else vis_emb
            kv = vis_emb
            fused = self.fusion(q, kv)
        else:
            fused = self.fusion(*embeddings)

        # --- Classify ---
        logits = self.classifier(fused)

        return logits, fused
