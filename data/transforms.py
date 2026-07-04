"""
数据增强模块 — 图像 / 传感器时序 / 文本。
已修复 Bbox 漂移、NoneType 崩溃、双重归一化及长文本匹配问题。
"""
from __future__ import annotations

import random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ======================================================================
# 图像增强 (Albumentations)
# ======================================================================

def get_image_transforms(
    image_size: tuple[int, int] = (640, 640),
    augment: bool = False,
) -> A.Compose:
    """获取图像变换管线。"""

    # 🔥 核心修复 1：定义统一的 Bbox 坐标转换规则（YOLO 格式: [x_center, y_center, w, h]）
    bbox_params = A.BboxParams(
        format='yolo',
        label_fields=['class_labels'], # 告诉库，类别标签是哪个变量
        min_visibility=0.3             # 如果裁剪后框的可视面积不到 30%，直接丢弃该框，防止假阳性
    )

    if augment:
        return A.Compose([
            A.RandomResizedCrop(
                height=image_size[0], width=image_size[1],
                scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0,
            ),
            A.Rotate(limit=30, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.MotionBlur(blur_limit=(3, 7), p=0.5),
            ], p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, border_mode=0, p=0.3),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(), # 🔥 核心修复 3：直接输出 PyTorch Tensor，且处理好 Channel 通道
        ], bbox_params=bbox_params)
    else:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=bbox_params)


# ======================================================================
# 时序传感器增强
# ======================================================================

def sensor_augment(
    x: torch.Tensor | None,
    jitter_std: float = 0.01,
    scale_range: tuple[float, float] = (0.95, 1.05),
    mask_prob: float = 0.1,
    mask_value: float = 0.0,
) -> torch.Tensor | None:
    """对传感器时序数据做轻量增强。"""
    # 🔥 核心修复 2：拦截 None，完美配合之前写好的显存优化
    if x is None:
        return None

    if x.ndim == 2:
        x = x.unsqueeze(0)
        was_2d = True
    else:
        was_2d = False

    B, C, T = x.shape

    if jitter_std > 0:
        noise_std = jitter_std * x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        x = x + torch.randn_like(x) * noise_std

    lo, hi = scale_range
    scale = torch.empty(B, C, 1).uniform_(lo, hi)
    x = x * scale

    if mask_prob > 0:
        mask = torch.rand(B, 1, T) > mask_prob
        x = x * mask.float() + mask_value * (~mask).float()

    if was_2d:
        x = x.squeeze(0)

    return x


# ======================================================================
# 文本增强
# ======================================================================

_RAW_KEYWORDS = {
    "滑坡", "塌陷", "落石", "倒树", "裂缝", "位移",
    "降雨", "暴雨", "水位", "孔隙水压", "变形", "岩体",
    "泥石流", "坡面", "路基", "路面", "裂缝宽度", "风险",
    "预警", "应急", "巡检", "施工", "安全帽", "作业",
}
# 🔥 核心修复 4：按长度降序排序！确保“裂缝宽度”优先于“裂缝”被匹配
GEOLOGY_KEYWORDS = sorted(list(_RAW_KEYWORDS), key=len, reverse=True)

def text_augment(
    text: str,
    drop_prob: float = 0.05,
    shuffle_window: int = 3,
) -> str:
    """轻度文本增强：随机删除非关键词。"""
    if not text or drop_prob <= 0:
        return text

    chars = list(text)
    i = 0
    while i < len(chars):
        in_keyword = False
        for kw in GEOLOGY_KEYWORDS:
            end = i + len(kw)
            if end <= len(chars) and "".join(chars[i:end]) == kw:
                in_keyword = True
                i = end
                break

        if in_keyword:
            continue

        if random.random() < drop_prob:
            chars.pop(i)
        else:
            i += 1

    if len(chars) == 0:
        return text

    return "".join(chars)