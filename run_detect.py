#!/usr/bin/env python
"""
YOLO best.pt 检测测试 — 在 test 数据集上运行并对比真实标注。

用法（必须在项目根目录运行）:
    cd E:\transform\geohazard
    python run_detect.py
"""
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from easydict import EasyDict

# ============================================================
# 🔧 关键修复：抑制 ultralytics YOLO 加载时的训练/验证界面输出
# ============================================================
# 必须在 import ultralytics 之前设置
os.environ["ULTRALYTICS_VERBOSE"] = "0"        # 禁用 ultralytics 详细日志
os.environ["YOLO_VERBOSE"] = "True"            # 禁用 YOLO 详细输出
# 如果当前环境没有 CUDA 但配置了 device=cuda，强制 YOLO 不报错
# （ultralytics 内部会自动 fallback 到 CPU）
# 根据你的截图，run_detect.py 就在项目根目录，所以这里取 .parent 是正确的
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.predictor import GeoHazardPredictor

CLASS_NAMES = ["fallen tree", "landslide", "road collapse", "stone"]


def yolo_to_xyxy(xc, yc, w, h, W=640, H=640):
    """YOLO 归一化坐标 → 像素 xyxy"""
    return [(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H]


def compute_iou(a, b):
    x1 = max(a[0], b[0]);
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]);
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def load_gt(label_path):
    """读取 YOLO 标注 → [(class_id, [x1,y1,x2,y2]), ...]"""
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 5:
                cls_id = int(float(p[0]))
                boxes.append((cls_id, yolo_to_xyxy(*map(float, p[1:5]))))
    return boxes


def main():
    print("=" * 60)
    print("  YOLO best.pt 检测测试 — test 数据集")
    print("=" * 60)
    print(f"  项目根目录: {PROJECT_ROOT}")

    # 初始化 predictor（假设配置文件在 configs/ 目录下）
    print("\n加载模型...")
    p = GeoHazardPredictor(config_path=str(PROJECT_ROOT / "configs/default.yaml"))

    # 根据截图确定的绝对路径
    test_img_dir = PROJECT_ROOT / "data" / "raw" / "test" / "images"
    test_lbl_dir = PROJECT_ROOT / "data" / "raw" / "test" / "labels"

    print(f"  图像目录查找路径: {test_img_dir}")
    print(f"  标注目录查找路径: {test_lbl_dir}")

    # 1. 检查文件夹是否存在
    if not test_img_dir.exists():
        print(f"\n  ❌ 致命错误：找不到图像文件夹！请确认路径：{test_img_dir}")
        return

    # 2. 🔥 核心修复：无视大小写，抓取所有常见图片后缀 🔥
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [f for f in test_img_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
    images = sorted(images)

    print(f"  Test 图像总数: {len(images)}")

    if len(images) == 0:
        print("\n  ❌ 文件夹找到了，但里面没有图片！")
        print(f"  请打开电脑文件夹检查：{test_img_dir}")
        return

    # 全量测试逻辑
    gt_total = 0
    pred_total = 0
    matched_total = 0
    per_cls_gt = [0, 0, 0, 0]
    per_cls_pred = [0, 0, 0, 0]
    per_cls_match = [0, 0, 0, 0]
    img_count = 0

    for img_path in images:
        lbl_path = test_lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img_count += 1
        gt_boxes = load_gt(str(lbl_path))
        dets = p.detect(str(img_path), conf_threshold=0.25)

        for cls_id, _ in gt_boxes:
            per_cls_gt[cls_id] += 1
        gt_total += len(gt_boxes)

        for d in dets:
            per_cls_pred[d["class_id"]] += 1
        pred_total += len(dets)

        # 贪心 IoU 匹配
        matched_gt_idx = set()
        for d in dets:
            best_iou, best_idx = 0, -1
            for i, (gt_cls, gt_xyxy) in enumerate(gt_boxes):
                if i in matched_gt_idx:
                    continue
                iou = compute_iou(d["bbox"], gt_xyxy)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= 0.5 and d["class_id"] == gt_boxes[best_idx][0]:
                matched_gt_idx.add(best_idx)
                per_cls_match[d["class_id"]] += 1
        matched_total += len(matched_gt_idx)

    print(f"\n  实际处理有标注的图像: {img_count} 张")

    # 打印结果
    print("\n" + "=" * 60)
    print("  检测结果汇总")
    print("=" * 60)
    print(f"  真实目标 (GT):   {gt_total}")
    print(f"  预测目标 (Pred): {pred_total}")
    print(f"  匹配成功:        {matched_total}")

    print(f"\n  {'类别':<16} {'GT':>6} {'Pred':>6} {'匹配':>6} {'Recall':>8} {'Precision':>10}")
    print("  " + "-" * 56)

    for cls_id in range(4):
        gt = per_cls_gt[cls_id]
        pr = per_cls_pred[cls_id]
        mt = per_cls_match[cls_id]
        r = mt / gt if gt else 0
        prec = mt / pr if pr else 0
        print(f"  {CLASS_NAMES[cls_id]:<16} {gt:>6} {pr:>6} {mt:>6} {r:>7.1%} {prec:>9.1%}")

    overall_r = matched_total / gt_total if gt_total else 0
    overall_p = matched_total / pred_total if pred_total else 0
    f1 = 2 * overall_r * overall_p / (overall_r + overall_p) if (overall_r + overall_p) else 0

    print("  " + "-" * 56)
    print(f"  {'整体':<16} {gt_total:>6} {pred_total:>6} {matched_total:>6} {overall_r:>7.1%} {overall_p:>9.1%}")
    print(f"\n  F1-score: {f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    os.chdir(str(PROJECT_ROOT))
    main()
