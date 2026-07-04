"""
多模态 DataLoader — Collate Function + DataLoader 工厂。

处理 variable-length YOLO bboxes、可选传感器/报告字段。
"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


from __future__ import annotations

from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset


def multimodal_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """多模态 collate：处理不同样本的 variable-size 字段。

    - image:      stack → [B, 3, H, W]
    - sensor:     stack → [B, C, T] 或 None（彻底释放显存）
    - report:     list of str（不 stack）
    - label:      stack → [B]
    - bboxes:     list of [N_i, 5]（variable-size，保持 list）
    - image_path: list of str
    - label_name: list of str
    """
    collated: dict[str, Any] = {}

    for key in batch[0].keys():
        values = [item[key] for item in batch]

        if key == "image":
            collated[key] = torch.stack(values, dim=0)

        elif key == "sensor":
            # 🔥 如果没有传感器，坚决不造零张量，直接传 None 帮 GPU 减负
            if values[0] is None:
                collated[key] = None
            else:
                collated[key] = torch.stack(values, dim=0)

        elif key == "label":
            collated[key] = torch.stack([v for v in values], dim=0)

        elif key == "bboxes":
            # 保持 list，每个样本的 bbox 数量不同
            collated[key] = values

        elif key in ("report", "image_path", "label_name"):
            collated[key] = values

        else:
            # 其他未知键，保持 list
            collated[key] = values

    return collated


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    **kwargs,
) -> DataLoader:
    """构建多模态 DataLoader。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=multimodal_collate_fn,
        **kwargs,
    )