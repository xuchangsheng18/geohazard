#!/usr/bin/env python
"""
YOLO11 地质灾害检测模型训练 — 使用你的 4 类自定义数据集。

类别: 0: fallen tree, 1: landslide, 2: road collapse, 3: stone

用法（必须在项目根目录运行）:
    cd E:\transform\geohazard
    python train_yolo.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

# ─── 配置 ───────────────────────────────────────────
DATA_YAML  = "data/data.yaml"       # 你的 4 类数据集配置
PRETRAINED = "weights/best.pt"           # 从官方预训练权重开始（首次自动下载）
EPOCHS     = 50
IMG_SIZE   = 640
BATCH      = 8
DEVICE     = 0                      # GPU 编号，None 或 "cpu" 表示 CPU
PROJECT    = "runs/detect"          # 输出根目录
NAME       = "geohazard"            # 本次训练的目录名
# ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  YOLO11 地质灾害检测训练")
    print(f"  数据集:      {Path(DATA_YAML).resolve()}")
    print(f"  预训练权重:  {PRETRAINED}")
    print(f"  Epochs:      {EPOCHS}")
    print(f"  Batch:       {BATCH}")
    print(f"  Image Size:  {IMG_SIZE}")
    print("=" * 60)

    # 加载预训练模型
    model = YOLO(PRETRAINED, task='detect', verbose=False)

    # 训练 — 直接复用 ultralytics 官方训练流程
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=0.0005,
        patience=20,             # 20 epoch 不提升则早停
        cos_lr=True,
        plots=True,              # 生成训练曲线图
        save=True,
        val=True,                # 每个 epoch 之后验证
    )

    # 复制最优模型到 weights/ 目录
    best_src = Path(PROJECT) / NAME / "weights" / "best.pt"
    best_dst = PROJECT_ROOT / "weights" / "best.pt"
    if best_src.exists():
        import shutil
        best_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(best_src), str(best_dst))
        print(f"\n✓ 最优模型已复制到: {best_dst}")
    else:
        print(f"\n✗ 未找到最优模型文件: {best_src}")
