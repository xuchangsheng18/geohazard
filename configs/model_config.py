"""
Typed configuration dataclasses for the GeoHazard multimodal system.
All configs can be loaded from YAML via OmegaConf and validated here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class VisionConfig:
    model_name: str = "yolo11n.pt"
    custom_weights: str = "weights/best.pt"
    data_yaml: str = "data/data.yaml"
    input_size: tuple[int, int] = (640, 640)
    feature_dim: int = 512
    freeze_backbone: bool = True


@dataclass
class TimeSeriesConfig:
    enabled: bool = False
    input_channels: int = 6
    seq_length: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    feature_dim: int = 256


@dataclass
class TextConfig:
    enabled: bool = False
    model_name: str = "bert-base-chinese"
    max_length: int = 512
    feature_dim: int = 256
    freeze_backbone: bool = True


@dataclass
class CrossAttentionConfig:
    query_modality: str = "timeseries"
    key_value_modality: str = "vision"


@dataclass
class FusionConfig:
    method: Literal["concat", "gated", "cross_attention", "transformer"] = "concat"
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    cross_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    warmup_steps: int = 1000
    lr_scheduler: str = "cosine"
    early_stopping_patience: int = 10
    grad_clip_norm: float = 1.0


@dataclass
class DataConfig:
    root: str = "./data/raw"
    yolo_config: str = "data/data.yaml"
    sensor_dir: str = "sensors"      # resolved under {root}/{split}/sensors/
    report_dir: str = "reports"      # resolved under {root}/{split}/reports/
    num_workers: int = 4


@dataclass
class GeoHazardConfig:
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "./outputs"
    num_classes: int = 4   # 0:fallen tree  1:landslide  2:road collapse  3:stone
    class_names: list[str] = field(default_factory=lambda: [
        "fallen tree", "landslide", "road collapse", "stone"
    ])
    mixed_precision: str = "fp16"
    vision: VisionConfig = field(default_factory=VisionConfig)
    timeseries: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)
    text: TextConfig = field(default_factory=TextConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
