"""
Inference pipeline for the GeoHazard multimodal detection system.

Supports:
  - Single-sample prediction (image + optional sensor + optional text)
  - YOLO detection on raw images (using your best.pt)
  - Batch prediction from CSV
  - Feature extraction (for downstream analysis or retrieval)

Usage (Python API):
    predictor = GeoHazardPredictor("configs/default.yaml", "outputs/best_model.pt")
    result = predictor.predict(image_path="data/raw/test/images/img_001.jpg",
                               sensor_path="data/raw/test/sensors/img_001.npy",
                               report_text="现场勘查：山体出现裂缝...")
    print(result["class_name"], result["confidence"])

Usage (CLI):
    python -m inference.predictor \
        --image ./data/raw/test/images/img_001.jpg \
        --sensor ./data/raw/test/sensors/sensor_001.npy \
        --report "现场报告内容..."
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional
from easydict import EasyDict
import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import OmegaConf

# Add project root to path
"""
Inference pipeline for the GeoHazard multimodal detection system.

Supports:
  - Single-sample prediction (image + optional sensor + optional text)
  - YOLO detection on raw images (using your best.pt)
  - Batch prediction from CSV
  - Feature extraction (for downstream analysis or retrieval)

Usage (Python API):
    predictor = GeoHazardPredictor("configs/default.yaml", "outputs/best_model.pt")
    result = predictor.predict_full(image_path="data/raw/test/images/img_001.jpg",
                                    report_text="现场勘查：几名工人正在施工...")
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional
from easydict import EasyDict
import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.model_config import GeoHazardConfig
from models.geohazard_model import GeoHazardMultimodalModel


# 🔥 修改点 1：补齐第 5 类 safe
DEFAULT_CLASS_NAMES = ["fallen tree", "landslide", "road collapse", "stone", "safe"]


class GeoHazardPredictor:
    """多模态地质灾害预测器。"""

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # --- 加载配置 ---
        cfg_dict = OmegaConf.load(config_path)
        cfg_dict = OmegaConf.to_container(cfg_dict, resolve=True)
        self.config = EasyDict(cfg_dict)

        self.device = device or self.config.device
        self.input_size = self.config.vision["input_size"]

        if hasattr(self.config, "class_names") and self.config.class_names:
            self.class_names = self.config.class_names
        else:
            self.class_names = DEFAULT_CLASS_NAMES
        logger.info(f"灾害类别 ({len(self.class_names)}): {self.class_names}")

        # --- 构建多模态模型 ---
        logger.info("Building GeoHazardMultimodalModel...")
        self.model = GeoHazardMultimodalModel(self.config)

        # --- 加载融合模型 checkpoint ---
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            logger.info(f"Loading fusion checkpoint: {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"], strict=False)
            saved_names = state.get("class_names", None)
            if saved_names:
                self.class_names = saved_names
        else:
            logger.warning("未加载融合 checkpoint，融合层使用随机初始化。")

        self.model.to(self.device)
        self.model.eval()
        logger.info("Predictor ready.")

    # =====================================================================
    # 图像预处理
    # =====================================================================

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"图像不存在: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, tuple(self.input_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img.unsqueeze(0)

    def preprocess_sensor(self, sensor_path: str) -> torch.Tensor:
        if sensor_path.endswith(".npy"):
            arr = np.load(sensor_path)
        elif sensor_path.endswith(".csv"):
            arr = pd.read_csv(sensor_path).values.T
        else:
            raise ValueError(f"不支持的传感器格式: {sensor_path}")

        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        seq_len = self.config.timeseries["seq_length"]
        ch, t = arr.shape
        if t < seq_len:
            pad = np.zeros((ch, seq_len - t), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=1)
        else:
            arr = arr[:, :seq_len]

        return torch.from_numpy(arr).float().unsqueeze(0)

    # =====================================================================
    # YOLO 目标检测
    # =====================================================================

    @torch.no_grad()
    def detect(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> list[dict]:
        yolo = self.model.vision_encoder.yolo
        results = yolo.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            show=False,
            save=False,
            stream=False,
        )

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls.item())
                    # 只有前4类(真实灾害)才允许画框输出，safe类本身就没有框
                    if cls_id < 4:
                        detections.append({
                            "class_id": cls_id,
                            "class_name": self.class_names[cls_id],
                            "confidence": round(box.conf.item(), 4),
                            "bbox": box.xyxy[0].tolist(),
                        })
        return detections

    # =====================================================================
    # 多模态分类预测
    # =====================================================================

    @torch.no_grad()
    def predict(
        self,
        image_path: Optional[str] = None,
        image_tensor: Optional[torch.Tensor] = None,
        sensor_path: Optional[str] = None,
        sensor_tensor: Optional[torch.Tensor] = None,
        report_text: Optional[str] = None,
        return_embedding: bool = False,
    ) -> dict:
        if image_tensor is not None:
            images = image_tensor.to(self.device)
        elif image_path is not None:
            images = self.preprocess_image(image_path).to(self.device)
        else:
            raise ValueError("必须提供 image_path 或 image_tensor。")

        sensors = None
        if sensor_tensor is not None:
            sensors = sensor_tensor.to(self.device)
        elif sensor_path is not None and os.path.exists(sensor_path):
            sensors = self.preprocess_sensor(sensor_path).to(self.device)

        reports = [report_text] if report_text is not None else [""]

        logits, embedding = self.model(images, sensors, reports)
        probs = torch.softmax(logits, dim=-1)
        pred_id = probs.argmax(dim=-1).item()
        confidence = probs[0, pred_id].item()

        result = {
            "class_id": pred_id,
            "class_name": self.class_names[pred_id] if pred_id < len(self.class_names) else f"class_{pred_id}",
            "confidence": round(confidence, 4),
            "probs": {
                self.class_names[i] if i < len(self.class_names) else f"class_{i}": round(p, 4)
                for i, p in enumerate(probs[0].tolist())
            },
        }

        if return_embedding:
            result["embedding"] = embedding.cpu().numpy()

        return result

    # =====================================================================
    # 综合预测（YOLO 检测 + 多模态分类）
    # =====================================================================
    @torch.no_grad()
    def predict_full(
            self,
            image_path: str,
            sensor_path: Optional[str] = None,
            report_text: Optional[str] = None,
            conf_threshold: float = 0.25,
    ) -> dict:
        """多模态深度纠偏：结合模型输出的安全概率，实现真正的智能压制。"""

        # 1. 优先执行多模态分类，获取全局智能判定
        cls_result = self.predict(
            image_path=image_path,
            sensor_path=sensor_path,
            report_text=report_text,
            return_embedding=False,
        )

        # 🔥 修改点 2：直接读取模型给出的 safe 概率和 stone 概率
        stone_prob = cls_result["probs"].get("stone", 0.0)
        safe_prob = cls_result["probs"].get("safe", 0.0)

        # 如果安全概率极高 (>0.6)，说明多模态模型非常有把握这就是工人/安全背景
        # 我们直接对检测器的底线要求变得极其苛刻
        current_conf_threshold = 0.6 if safe_prob > 0.6 else conf_threshold

        # 2. 执行目标检测
        detections = self.detect(image_path, conf_threshold=current_conf_threshold)

        # 3. 🔥 语义置信度加权：降维打击误报 🔥
        refined_detections = []
        for d in detections:
            if d["class_name"] == "stone":
                # 新逻辑：石头概率越低，安全概率越高，对 YOLO 框的抹杀越狠
                # 假设 YOLO 报了 0.8 的置信度，但 safe_prob 为 0.9
                # 惩罚系数 = (0.1^0.5) * (1 - 0.9) = 0.31 * 0.1 = 0.03
                # 最终框置信度直接被干碎成 0.8 * 0.03 = 0.024，直接剔除！
                penalty_factor = (stone_prob ** 0.5) * (1.0 - safe_prob)
                d["confidence"] = round(d["confidence"] * penalty_factor, 4)

                if d["confidence"] < conf_threshold:
                    continue

            refined_detections.append(d)

        return {
            "detections": refined_detections,
            "classification": cls_result,
            "image_path": image_path,
        }

    # =====================================================================
    # 批量预测与特征提取 (保持原样)
    # =====================================================================
    # ... (此处省略 predict_batch 和 extract_features，因其逻辑不变) ...


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GeoHazard Multimodal Predictor")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pt", help="融合模型 checkpoint")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--sensor", type=str, default=None, help="传感器数据路径")
    parser.add_argument("--report", type=str, default=None, help="中文现场报告文本")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["classify", "detect", "full"],
                        help="推理模式: classify(多模态分类), detect(YOLO检测), full(全部)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 检测置信度阈值")
    args = parser.parse_args()

    predictor = GeoHazardPredictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )

    if args.mode == "detect":
        dets = predictor.detect(args.image, conf_threshold=args.conf)
        print(f"\n检测到 {len(dets)} 个目标:")
        for d in dets:
            print(f"  [{d['class_name']}] conf={d['confidence']:.3f}  bbox={d['bbox']}")

    elif args.mode == "full":
        result = predictor.predict_full(
            image_path=args.image,
            sensor_path=args.sensor,
            report_text=args.report,
            conf_threshold=args.conf,
        )
        print("\n" + "=" * 50)
        print("  综合预测结果 (多模态联合修正后)")
        print("=" * 50)
        print(f"\n  [YOLO 检测] {len(result['detections'])} 个目标:")
        for d in result["detections"]:
            print(f"    [{d['class_name']}] conf={d['confidence']:.3f}")
        print(f"\n  [多模态分类 (顶层大脑)]")
        print(f"    预测主类别: {result['classification']['class_name']}")
        print(f"    最高置信度: {result['classification']['confidence']:.4f}")
        print(f"    全类别概率: {result['classification']['probs']}")

    else:
        result = predictor.predict(
            image_path=args.image,
            sensor_path=args.sensor,
            report_text=args.report,
            return_embedding=True,
        )
        print("\n" + "=" * 50)
        print("  多模态分类结果")
        print("=" * 50)
        print(f"  类别:      {result['class_name']}")
        print(f"  置信度:    {result['confidence']:.4f}")
        print(f"  各类概率:  {result['probs']}")
        print("=" * 50)