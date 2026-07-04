"""
Multimodal dataset loader for geological hazard detection.

直接扫描 YOLO 目录结构，自动发现所有图像和标注文件。
图像级分类标签从 YOLO 标注文件中自动推断（取多数类）。

数据目录结构 (root = data/raw/):
    data/raw/
    ├── data.yaml                  # YOLO 数据集描述（类别名、路径）
    │
    ├── train/                     # YOLO 训练集
    │   ├── images/                #   原图 (.jpg / .png)
    │   ├── labels/                #   YOLO 标注 (.txt, 同名)
    │   ├── sensors/               #   (可选) 传感器时序数据 (.npy/.csv)
    │   └── reports/               #   (可选) 文本报告 (.txt)
    ├── valid/                     # YOLO 验证集
    │   ├── images/
    │   ├── labels/
    │   ├── sensors/
    │   └── reports/
    └── test/                      # YOLO 测试集
        ├── images/
        ├── labels/
        ├── sensors/
        └── reports/

不需要 labels.csv — 图像级标签直接从 YOLO 标注中取每张图中出现最多的类别。
传感器和文本通过文件名映射（与图像同名、不同扩展名、不同目录）。
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from loguru import logger

# 💡 引入上一轮修改好的增强逻辑
from data.transforms import get_image_transforms


class GeoHazardDataset(Dataset):
    """多模态地质灾害数据集加载器（支持负样本与 Albumentations 边界框同步联动增强）。"""

    # 新增第 5 类 "safe" (索引为 4)
    CLASS_NAMES = ["fallen tree", "landslide", "road collapse", "stone", "safe"]

    def __init__(
            self,
            root: str,
            sensor_dir: str = "sensors",
            report_dir: str = "reports",
            split: str = "train",
            image_size: tuple[int, int] = (640, 640),
            sensor_seq_len: int = 256,
            text_max_len: int = 512,
            augment: bool = False,
            image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
            seed: int = 42,
    ):
        self.root = Path(root)
        self.sensor_dir = self.root / split / sensor_dir
        self.report_dir = self.root / split / report_dir
        self.image_size = image_size
        self.sensor_seq_len = sensor_seq_len
        self.text_max_len = text_max_len
        self.augment = augment
        self.split = split

        # ---------------------------------------------------------------
        # 类别与数据变换初始化
        # ---------------------------------------------------------------
        self.label2id = {name: i for i, name in enumerate(self.CLASS_NAMES)}
        self.id2label = {i: name for name, i in self.label2id.items()}
        logger.info(f"类别 ({len(self.CLASS_NAMES)}): {self.label2id}")

        # 🔥 核心修改 1：实例化上一轮我们写好的、带 Bbox 同步保护的变换管线
        self.transform = get_image_transforms(image_size=self.image_size, augment=self.augment)

        # ---------------------------------------------------------------
        # 扫描 {split}/images/ 目录，建立样本列表
        # ---------------------------------------------------------------
        images_dir = self.root / split / "images"
        labels_dir = self.root / split / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(
                f"图像目录不存在: {images_dir}\n"
                f"预期的 YOLO 目录结构: data/raw/{split}/images/"
            )

        self.samples: list[dict] = []
        skipped_no_label = 0

        for ext in image_extensions:
            for img_path in sorted(images_dir.glob(f"*{ext}")):
                lbl_path = labels_dir / (img_path.stem + ".txt")

                if lbl_path.exists():
                    bboxes = self._read_yolo_labels(str(lbl_path))
                    img_label_id = self._majority_class(bboxes)
                else:
                    skipped_no_label += 1
                    bboxes = torch.zeros(0, 5)
                    img_label_id = 4  # 如果没有标签文件（负样本），直接归为 safe 类 (4)

                self.samples.append({
                    "image_path": str(img_path),
                    "label_path": str(lbl_path) if lbl_path.exists() else None,
                    "image_rel": f"{split}/images/{img_path.name}",
                    "image_stem": img_path.stem,
                    "label": img_label_id,
                    "bboxes": bboxes,
                })

        logger.info(
            f"Split [{split}]: 扫描到 {len(self.samples)} 张图像"
            + (f", {skipped_no_label} 张无标注(已判定为 safe)" if skipped_no_label > 0 else "")
        )

        if len(self.samples) == 0:
            raise RuntimeError(f"Split [{split}] 未找到任何图像！请检查 {images_dir}")

        # 统计类别分布
        class_counts = {}
        for s in self.samples:
            cls_name = self.id2label[s["label"]]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        logger.info(f"  类别分布: {class_counts}")

        self._has_sensors = self.sensor_dir.exists() and any(self.sensor_dir.iterdir())
        self._has_reports = self.report_dir.exists() and any(self.report_dir.iterdir())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int | list | None]:
        s = self.samples[idx]

        sample: dict = {
            "label": torch.tensor(s["label"], dtype=torch.long),
            "label_name": self.id2label[s["label"]],
            "image_path": s["image_rel"],
        }

        # ---------------------------------------------------------------
        # 1. 图像加载与 Albumentations 联动联动增强
        # ---------------------------------------------------------------
        img = cv2.imread(s["image_path"])
        if img is None:
            # 防御性读取失败处理
            sample["image"] = torch.zeros(3, *self.image_size)
            sample["bboxes"] = torch.zeros(0, 5)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 🔥 核心修改 2：把 YOLO 格式的 (N, 5) 拆解成 Albumentations 要求的输入格式
            bboxes_tensor = s["bboxes"].clone()
            if bboxes_tensor.shape[0] > 0:
                # 只有属于灾害类别（前4类）的框才参与空间变换同步
                boxes_list = bboxes_tensor[:, 1:].tolist()  # [[x_c, y_c, w, h], ...]
                class_labels = bboxes_tensor[:, 0].long().tolist()  # [class_id, ...]
            else:
                boxes_list = []
                class_labels = []

            # 🔥 核心修改 3：送入管线，同步对图像像素和边界框进行翻转、裁剪、旋转
            transformed = self.transform(image=img, bboxes=boxes_list, class_labels=class_labels)

            # 经过 ToTensorV2 变换后，这里已经是 [3, H, W] 且归一化好的 Tensor 了！
            sample["image"] = transformed["image"]

            # 🔥 核心修改 4：增强完毕后，重新把变动过坐标的边界框组装回 (N, 5) 的 Tensor 格式
            if len(transformed["bboxes"]) > 0:
                new_boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
                new_labels = torch.tensor(transformed["class_labels"], dtype=torch.float32).unsqueeze(1)
                sample["bboxes"] = torch.cat([new_labels, new_boxes], dim=1)
            else:
                sample["bboxes"] = torch.zeros(0, 5)

        # ---------------------------------------------------------------
        # 2. 传感器与文本报告匹配
        # ---------------------------------------------------------------
        sample["sensor"] = self._load_sensor_by_stem(s["image_stem"])
        sample["report"] = self._load_report_by_stem(s["image_stem"])

        return sample

    # ==================================================================
    # 内部方法
    # ==================================================================

    @staticmethod
    def _read_yolo_labels(path: str) -> torch.Tensor:
        boxes = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    boxes.append([float(p) for p in parts[:5]])
        if not boxes:
            return torch.zeros(0, 5)
        return torch.tensor(boxes, dtype=torch.float32)

    def _majority_class(self, bboxes: torch.Tensor) -> int:
        if bboxes.shape[0] == 0:
            return 4

        class_ids = bboxes[:, 0].long()
        valid_mask = class_ids < (len(self.CLASS_NAMES) - 1)
        class_ids = class_ids[valid_mask]

        if class_ids.shape[0] == 0:
            return 4

        counts = torch.bincount(class_ids, minlength=len(self.CLASS_NAMES))
        return counts.argmax().item()

    def _load_sensor_by_stem(self, stem: str) -> Optional[torch.Tensor]:
        if not self._has_sensors:
            return None
        for ext in [".npy", ".csv"]:
            sp = self.sensor_dir / (stem + ext)
            if sp.exists():
                return self._load_sensor(str(sp))
        return None

    def _load_sensor(self, path: str) -> torch.Tensor:
        if path.endswith(".npy"):
            arr = np.load(path)
        elif path.endswith(".csv"):
            import pandas as pd
            arr = pd.read_csv(path).values.T
        else:
            raise ValueError(f"不支持的传感器格式: {path}")
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        ch, t = arr.shape
        if t < self.sensor_seq_len:
            pad = np.zeros((ch, self.sensor_seq_len - t), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=1)
        else:
            arr = arr[:, :self.sensor_seq_len]
        return torch.from_numpy(arr).float()

    def _load_report_by_stem(self, stem: str) -> str:
        if not self._has_reports:
            return ""
        rp = self.report_dir / (stem + ".txt")
        if rp.exists():
            return self._load_report(str(rp))
        return ""

    def _load_report(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()