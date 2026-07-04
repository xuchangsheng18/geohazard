"""
Text encoder for geological field reports.

Design rationale:
  We use a pre-trained BERT/RoBERTa (via HuggingFace Transformers) as the
  backbone for encoding unstructured Chinese text reports from field surveys.

  The [CLS] token embedding serves as the sentence-level representation,
  projected to a unified feature dimension for multimodal fusion.

Reference:
  - Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers
    for Language Understanding." NAACL 2019.
  - Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach." 2019.
  - For Chinese geological texts specifically, models like
    "hfl/chinese-roberta-wwm-ext" or "bert-base-chinese" are recommended.
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from loguru import logger
from transformers import AutoModel, AutoTokenizer
import os

class TextEncoder(nn.Module):
    """BERT-based text encoder for geological field reports.

    Usage::

        encoder = TextEncoder(model_name="bert-base-chinese", feature_dim=256)
        embedding = encoder(["山体出现裂缝，宽度约5厘米..."])  # [B, 256]
    """

    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        max_length: int = 512,
        feature_dim: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.max_length = max_length
        project_root = Path(__file__).resolve().parent.parent

        # 2. 尝试拼接本地 weights 路径
        local_path = project_root / "weights" / model_name
        if local_path.exists():
            logger.info(f"检测到本地模型，直接加载: {local_path}")
            model_path = str(local_path)
        else:
            logger.info(f"本地未找到模型，将尝试在线下载: {model_name}")
            model_path = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.backbone = AutoModel.from_pretrained(model_path)

        if freeze_backbone:
            logger.info("Freezing BERT backbone.")
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size  # usually 768

        # Project BERT's [CLS] embedding to the unified fusion dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
        )

    @torch.no_grad()
    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Converts raw text list into tokenized inputs."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Args:
            texts: list[str] — batch of geological field reports

        Returns:
            embeddings: [B, feature_dim]
        """
        inputs = self.tokenize(texts)
        device = next(self.backbone.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.backbone(**inputs)
        # Use [CLS] token (first token) as sentence embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]
        return self.projection(cls_embedding)                # [B, feature_dim]
