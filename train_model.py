#!/usr/bin/env python
"""
Training script for the GeoHazard Multimodal Detection System.

数据集结构（YOLO 格式，无需 labels.csv）:
    data/raw/
    ├── data.yaml          # YOLO 配置（类别名、路径）
    ├── train/             # 训练集
    │   ├── images/        #   原图
    │   └── labels/        #   YOLO 标注 (.txt)
        └── reports/
    ├── valid/             # 验证集
    │   ├── images/
    │   └── labels/
    ├── test/              # 测试集
    │   ├── images/
    │   └── labels/
    ├── sensors/           # (可选) 传感器数据
    └── reports/           # (可选) 文本报告

类别 (4 类):
    0: fallen tree    1: landslide    2: road collapse    3: stone

Usage:
    python train_model.py --config configs/default.yaml
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import GeoHazardConfig
from models.geohazard_model import GeoHazardMultimodalModel
from data.dataset import GeoHazardDataset
from data.dataloader import build_dataloader
from utils.metrics import MetricsTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Train GeoHazard Multimodal Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint 路径")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: GeoHazardConfig,
    steps_per_epoch: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """构建 AdamW 优化器 + warmup + cosine 衰减调度器。"""

    vision_params = []
    sensor_params = []
    text_params = []
    fusion_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "vision" in name:
            vision_params.append(param)
        elif "sensor" in name:
            sensor_params.append(param)
        elif "text" in name:
            text_params.append(param)
        elif "fusion" in name:
            fusion_params.append(param)
        elif "classifier" in name:
            classifier_params.append(param)

    param_groups = [
        {"params": vision_params, "lr": config.training.learning_rate * 0.1, "name": "vision"},
        {"params": sensor_params, "lr": config.training.learning_rate, "name": "sensor"},
        {"params": text_params, "lr": config.training.learning_rate, "name": "text"},
        {"params": fusion_params, "lr": config.training.learning_rate, "name": "fusion"},
        {"params": classifier_params, "lr": config.training.learning_rate, "name": "classifier"},
    ]

    optimizer = AdamW(
        param_groups,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    total_steps = config.training.epochs * steps_per_epoch
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=config.training.warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - config.training.warmup_steps,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.training.warmup_steps],
    )

    return optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    metrics: MetricsTracker,
    device: torch.device,
    epoch: int,
    config: GeoHazardConfig,
) -> float:
    """训练一个 epoch，返回平均 loss。"""
    model.train()
    metrics.reset()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
    for batch in pbar:
        images = batch["image"].to(device)
        sensors = batch["sensor"].to(device) if batch["sensor"] is not None else None
        reports = batch["report"]
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        use_amp = config.mixed_precision == "fp16" and device.type == "cuda"
        with autocast(enabled=use_amp):
            logits, _ = model(images, sensors, reports)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        metrics.update(logits.detach(), labels)
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    metrics: MetricsTracker,
    device: torch.device,
    config: GeoHazardConfig,
) -> tuple[float, dict]:
    """验证，返回 (avg_loss, metrics_dict)。"""
    model.eval()
    metrics.reset()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Validation"):
        images = batch["image"].to(device)
        sensors = batch["sensor"].to(device) if batch["sensor"] is not None else None
        reports = batch["report"]
        labels = batch["label"].to(device)

        logits, _ = model(images, sensors, reports)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        metrics.update(logits, labels)

    avg_loss = total_loss / len(dataloader)
    metrics_dict = metrics.compute()
    return avg_loss, metrics_dict


def main():
    args = parse_args()

    # --- 配置 ---
    cfg_dict = OmegaConf.load(args.config)
    cfg_dict = OmegaConf.to_container(cfg_dict, resolve=True)
    config = GeoHazardConfig(**cfg_dict)
    device = torch.device(args.device or config.device)

    logger.info(f"Device: {device}")
    logger.info(f"类别数: {config.num_classes}, 类别: {config.class_names}")

    # --- 输出目录 ---
    os.makedirs(config.output_dir, exist_ok=True)

    # --- Wandb ---
    if args.use_wandb:
        import wandb
        wandb.init(project="geohazard-multimodal", config=cfg_dict)

    # ==================================================================
    # 数据集 — 直接从 YOLO 目录结构读取，按 train/ 和 valid/ 拆分
    # ==================================================================
    logger.info("加载训练集 (data/raw/train/) ...")
    train_dataset = GeoHazardDataset(
        root=config.data.root,
        sensor_dir=config.data.sensor_dir,
        report_dir=config.data.report_dir,
        split="train",
        image_size=config.vision.input_size,
        sensor_seq_len=config.timeseries.seq_length,
        text_max_len=config.text.max_length,
        augment=True,
        seed=config.seed,
    )

    logger.info("加载验证集 (data/raw/valid/) ...")
    val_dataset = GeoHazardDataset(
        root=config.data.root,
        sensor_dir=config.data.sensor_dir,
        report_dir=config.data.report_dir,
        split="valid",
        image_size=config.vision.input_size,
        sensor_seq_len=config.timeseries.seq_length,
        text_max_len=config.text.max_length,
        augment=False,
        seed=config.seed,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )

    val_loader = build_dataloader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    # ==================================================================
    # 模型
    # ==================================================================
    model = GeoHazardMultimodalModel(config).to(device)

    start_epoch = 0
    if args.resume:
        logger.info(f"从 {args.resume} 恢复训练")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1

    # --- 优化器 & 调度器 ---
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, len(train_loader))
    scaler = GradScaler(enabled=(config.mixed_precision == "fp16"))

    # --- 指标追踪 ---
    class_names = config.class_names if config.class_names else GeoHazardDataset.CLASS_NAMES
    train_metrics = MetricsTracker(class_names)
    val_metrics = MetricsTracker(class_names)

    # ==================================================================
    # 训练循环
    # ==================================================================
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(start_epoch, config.training.epochs):
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Epoch {epoch + 1} / {config.training.epochs}")
        logger.info(f"{'=' * 50}")

        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            scaler, train_metrics, device, epoch, config,
        )
        train_m = train_metrics.compute()
        logger.info(
            f"Train Loss: {train_loss:.4f} | "
            f"Acc: {train_m['accuracy']:.4f} | F1: {train_m['f1_macro']:.4f}"
        )

        # 验证
        val_loss, val_m = validate(model, val_loader, val_metrics, device, config)
        logger.info(
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {val_m['accuracy']:.4f} | F1: {val_m['f1_macro']:.4f} | "
            f"ROC-AUC: {val_m.get('roc_auc', 'N/A')}"
        )

        # Wandb 日志
        if args.use_wandb:
            wandb.log({
                "train/loss": train_loss,
                "train/acc": train_m["accuracy"],
                "train/f1": train_m["f1_macro"],
                "val/loss": val_loss,
                "val/acc": val_m["accuracy"],
                "val/f1": val_m["f1_macro"],
                "val/roc_auc": val_m.get("roc_auc", 0),
                "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

        # 保存最优模型
        val_f1 = val_m["f1_macro"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            ckpt_path = os.path.join(config.output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "val_f1": val_f1,
                    "class_names": class_names,
                    "config": config,
                },
                ckpt_path,
            )
            logger.info(f"✓ 保存最优模型 → {ckpt_path} (F1={val_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(
                f"未提升。早停计数: {patience_counter}/{config.training.early_stopping_patience}"
            )

        if patience_counter >= config.training.early_stopping_patience:
            logger.info("Early stopping 触发。")
            break

    # 绘制混淆矩阵
    val_metrics.plot_confusion_matrix(
        save_path=os.path.join(config.output_dir, "confusion_matrix.png")
    )

    logger.info(f"\n训练完成。最优 Val F1: {best_val_f1:.4f}")
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
