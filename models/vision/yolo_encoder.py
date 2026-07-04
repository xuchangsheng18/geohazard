"""
YOLO11-based visual encoder for geological hazard feature extraction.

Key design:
  - Loads your custom-trained `best.pt` via the Ultralytics SDK.
  - Hooks into an intermediate layer (e.g. neck output) to extract
    dense spatial features before the detection head.
  - Projects features to a fixed dim for downstream multimodal fusion.
  - Also exposes the raw YOLO object for detection inference.

你的 best.pt 对应的类别 (4 类):
    0: fallen tree    1: landslide    2: road collapse    3: stone

Reference (YOLO11): https://github.com/ultralytics/ultralytics
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from ultralytics import YOLO

# 与 data/data.yaml 保持一致的类别名
YOLO_CLASS_NAMES = ["fallen tree", "landslide", "road collapse", "stone"]


class YOLOFeatureExtractor(nn.Module):
    """Wraps a trained YOLO11 model and exposes its internal feature maps.

    Usage::

        extractor = YOLOFeatureExtractor(
            weights_path="weights/best.pt",
            feature_dim=512,
            freeze=True,
        )
        features = extractor(images)         # [B, 512, h, w]
        embedding = extractor.forward_embedding(images, pool="attention")  # [B, 512]
    """

    def __init__(
            self,
            weights_path: str,
            feature_dim: int = 512,
            input_size: tuple[int, int] = (640, 640),
            freeze: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.feature_dim = feature_dim

        # -----------------------------------------------------------------
        # STEP 1 — 加载 YOLO11，并给它穿上“隐身衣”！
        # -----------------------------------------------------------------
        logger.info(f"Loading YOLO11 weights from: {weights_path}")

        # 1. 建立一个临时的 YOLO 对象
        temp_yolo = YOLO(weights_path, task='detect', verbose=False)

        # 2. 🔥 双重保险 1：提取出纯净的 PyTorch 底层模型挂载
        self.model: nn.Module = temp_yolo.model

        # 3. 🔥 双重保险 2：把 YOLO 包装器塞进普通的 Python 列表里！
        # 这样 PyTorch 的 eval() 就绝对无法遍历到它，永远不会误触 .train()！
        self._yolo_container = [temp_yolo]

        # 读取 best.pt 中训练时的类别名
        if hasattr(temp_yolo, "names") and temp_yolo.names:
            self.detection_class_names = temp_yolo.names
        else:
            self.detection_class_names = YOLO_CLASS_NAMES

        # 👇 关键诊断：打印实际加载到的类别，一眼看出是 COCO 还是你的 4 类
        nc = len(self.detection_class_names) if isinstance(self.detection_class_names, dict) else len(self.detection_class_names)
        names_list = (list(self.detection_class_names.values())
                      if isinstance(self.detection_class_names, dict)
                      else list(self.detection_class_names))
        logger.info(f"YOLO 类别数: {nc}, 前5类: {names_list[:5]}")

        if nc > 4:
            logger.warning(
                f"⚠️ best.pt 包含 {nc} 类（应为 4 类地质灾害），"
                f"请确认权重文件是否用 data/data.yaml 训练！"
            )

        # -----------------------------------------------------------------
        # STEP 2 — 冻结 backbone（训练融合层时不更新 YOLO 参数）
        # -----------------------------------------------------------------
        if freeze:
            logger.info("Freezing YOLO backbone parameters.")
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        # -----------------------------------------------------------------
        # STEP 3 — 推断 YOLO 特征维度
        # -----------------------------------------------------------------
        self._raw_feature_dim = self._infer_feature_dim()
        logger.info(f"Inferred raw YOLO feature dim: {self._raw_feature_dim}")

        # -----------------------------------------------------------------
        # STEP 4 — 投影层：将 YOLO 特征映射到统一融合维度
        # -----------------------------------------------------------------
        self.projection = nn.Sequential(
            nn.Conv2d(self._raw_feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.SiLU(),
        )

    # 🔥 核心修复：通过动态属性暴露 yolo，既能让外层代码调用，又能骗过 PyTorch
    @property
    def yolo(self):
        """安全暴露 YOLO 对象，供 predictor.py 进行目标检测，同时避开 PyTorch 的扫描"""
        return self._yolo_container[0]

    def _infer_feature_dim(self) -> int:
        """通过 dummy forward 推断 YOLO 最后一层输出的通道数。"""
        dummy = torch.randn(1, 3, *self.input_size)
        with torch.no_grad():
            features = self._forward_features(dummy)
        return features.shape[1]

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取 YOLO 空间特征 (兼容 Ultralytics 的路由机制)。"""
        # 获取最底层的 sequential 模块列表
        core_model = self.model.model

        y = []  # 用于缓存每一层的输出，以支持 YOLO 的跳跃连接/路由逻辑

        for m in core_model:
            # 1. 处理 YOLO 特有的路由逻辑 (m.f)
            if hasattr(m, 'f') and m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]

            # 2. 前向传播
            x = m(x)

            # 3. 缓存输出
            y.append(x)

            # 4. 🔥 核心截断：在 SPPF 层截取高质量语义特征并立即返回
            if m.__class__.__name__ == 'SPPF':
                return x

        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W] 归一化到 [0, 1]

        Returns:
            features: [B, feature_dim, h, w] 空间特征图
        """
        x = self._forward_features(images)  # [B, C_raw, h, w]
        x = self.projection(x)  # [B, feature_dim, h, w]
        return x

    def forward_embedding(
            self, images: torch.Tensor, pool: str = "attention"
    ) -> torch.Tensor:
        """提取全局 embedding 向量（用于多模态融合）。"""
        features = self.forward(images)  # [B, D, h, w]
        if pool == "mean":
            return features.mean(dim=[2, 3])
        elif pool == "max":
            return features.amax(dim=[2, 3])
        elif pool == "attention":
            return self._attention_pool(features)
        else:
            raise ValueError(f"Unknown pool method: {pool}")

    def _attention_pool(self, x: torch.Tensor) -> torch.Tensor:
        """轻量空间注意力池化。"""
        B, C, H, W = x.shape
        attn = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        attn = attn.view(B, -1).softmax(dim=-1)  # [B, H*W]
        x_flat = x.view(B, C, -1)  # [B, C, H*W]
        pooled = (x_flat * attn.unsqueeze(1)).sum(-1)  # [B, C]
        return pooled