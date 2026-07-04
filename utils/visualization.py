"""
Visualization utilities for geohazard detection results.

Features:
  - Attention heatmap overlays (for interpretability)
  - YOLO detection prediction overlay on original images
  - Time-series + image side-by-side diagnostic views
  - Modality contribution bar charts (gated fusion)

灾害类别 (4 类):
    0: fallen tree    1: landslide    2: road collapse    3: stone
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 与 data.yaml 一致的 4 类
DEFAULT_CLASS_NAMES = ["fallen tree", "landslide", "road collapse", "stone"]

# 每类的可视化颜色 (BGR)
CLASS_COLORS = {
    0: (0, 255, 0),     # fallen tree    — 绿色
    1: (0, 165, 255),   # landslide      — 橙色
    2: (0, 0, 255),     # road collapse  — 红色
    3: (255, 0, 0),     # stone          — 蓝色
}


def overlay_predictions(
    image: np.ndarray,
    predictions: list[dict],
    class_names: Optional[list[str]] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """在图像上叠加 YOLO 检测框和类别标签。

    Args:
        image: RGB 图像 [H, W, 3], uint8
        predictions: [{"bbox": [x1,y1,x2,y2], "class_id": int, "confidence": float}, ...]
        class_names: 类别名列表，默认使用 4 类灾害名
        alpha: 检测框透明度

    Returns:
        annotated: [H, W, 3], uint8
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    img = image.copy()
    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred["bbox"])
        cls_id = int(pred["class_id"])
        conf = pred["confidence"]
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))

        # 半透明填充框
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

        # 边框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 标签
        label = f"{class_names[cls_id]}: {conf:.2f}" if cls_id < len(class_names) else f"cls_{cls_id}: {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def plot_attention_heatmap(
    image: np.ndarray,
    attention_map: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """将 Cross-Attention 热力图叠加到原始图像上。

    Args:
        image: [H, W, 3], uint8
        attention_map: [h, w] or [H, W], float
        save_path: 可选保存路径
    """
    if attention_map.shape[:2] != image.shape[:2]:
        attention_map = cv2.resize(attention_map, (image.shape[1], image.shape[0]))

    attn = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    heatmap = plt.cm.jet(attn)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlayed = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image)
    axes[0].set_title("原始图像")
    axes[1].imshow(attn, cmap="jet")
    axes[1].set_title("注意力热力图")
    axes[2].imshow(overlayed)
    axes[2].set_title("叠加效果")
    for ax in axes:
        ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_modality_contributions(
    gate_weights: np.ndarray,
    modality_names: Optional[list[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """门控融合的模态贡献权重柱状图。

    Args:
        gate_weights: [num_modalities] or [batch, num_modalities]
        modality_names: 模态名称，默认 ['Vision', 'Sensor', 'Text']
    """
    if modality_names is None:
        modality_names = ["Vision", "Sensor", "Text"]

    if gate_weights.ndim == 2:
        gate_weights = gate_weights.mean(axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    bars = ax.bar(modality_names, gate_weights, color=colors[: len(modality_names)])
    ax.set_ylabel("门控权重")
    ax.set_title("模态贡献度 (Gated Fusion)")
    ax.set_ylim(0, 1)

    for bar, val in zip(bars, gate_weights):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
