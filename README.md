# 🌍 GeoHazard-Multimodal — 多模态地质灾害检测系统

> 基于 **YOLO11** 视觉检测 + **Transformer** 多模态融合的端到端地质灾害智能预警平台

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3+-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 目录

- [项目背景](#项目背景)
- [灾害类别定义](#灾害类别定义)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [项目目录结构](#项目目录结构)
- [快速开始](#快速开始)
- [数据准备](#数据准备)
- [模型配置](#模型配置)
- [训练流程](#训练流程)
- [推理部署](#推理部署)
- [融合策略详解](#融合策略详解)
- [学术参考](#学术参考)
- [常见问题](#常见问题)

---

## 项目背景

地质灾害（滑坡、泥石流、崩塌、地裂缝等）的早期识别与预警是防灾减灾的核心难题。传统方法依赖单一数据源——要么只看遥感图像，要么只看传感器读数——信息孤岛导致漏报率和误报率居高不下。

本系统设计了一套**三模态融合**的深度学习框架：

| 模态 | 数据类型 | 编码器 | 作用 |
|------|---------|--------|------|
| 🖼️ **视觉** | RGB 图像 / 视频流 | **YOLO11** (`best.pt`) | 目标检测 + 空间特征提取 |
| 📈 **时序** | 传感器数据（降雨量、位移、孔隙水压等） | **PatchTST Transformer** | 时序模式建模：蠕变加速、水位骤升等预警信号 |
| 📝 **文本** | 现场调查报告、专家评估 | **BERT / RoBERTa** | 语义理解：提取报告中裂缝宽度、岩性、风险评级等关键信息 |

三路特征通过**可配置的多模态融合模块**（Cross-Attention / Gated Fusion / Transformer Fusion）聚合后，输出最终的地质灾害类别与置信度。

---

## 灾害类别定义

系统检测 **4 类** 地质灾害目标（与 YOLO 数据集一致）：

| ID | 类别名（英文） | 中文含义 | 典型视觉特征 |
|----|--------------|---------|-------------|
| 0 | `fallen tree` | 倒树 | 横倒的树干、断枝、连根拔起的树冠 |
| 1 | `landslide` | 滑坡 | 山体滑移面、裸露土石、坡面变形 |
| 2 | `road collapse` | 道路塌陷 | 路面坑洞、裂缝、路基下沉 |
| 3 | `stone` | 落石 | 散落或滚落的石块、岩体崩落 |

---

## 核心特性

- ✅ **即插即用 YOLO11**：通过 Ultralytics 官方 SDK 直接加载你已训练好的 `best.pt` 权重，无需重写网络结构
- ✅ **YOLO 数据集原生支持**：直接适配 `train/valid/test` + `images/labels` 的标准 YOLO 目录结构
- ✅ **检测 + 分类双能力**：同时提供 YOLO 目标检测（bbox）和多模态融合分类（image-level）
- ✅ **三模态全覆盖**：图像 + 传感器时序 + 文本报告，支持任意模态缺失（自动降级推理）
- ✅ **三种融合策略**：Cross-Attention（推荐）、Gated Fusion、Transformer Fusion，一行配置切换
- ✅ **工业级工程结构**：严格分层目录，数据处理 / 模型 / 融合 / 推理完全解耦
- ✅ **混合精度训练**：支持 FP16 AMP 训练，显存友好
- ✅ **完整评估体系**：Accuracy / Precision / Recall / F1 / ROC-AUC / 混淆矩阵
- ✅ **可解释性**：注意力热力图叠加、模态贡献权重可视化
- ✅ **灵活推理接口**：Python API + 命令行 CLI 三模式（classify / detect / full）

---

## 系统架构

```
                         ┌──────────────────────────────────────────────┐
                         │           🌍 GeoHazardMultimodalModel        │
                         │                                              │
   ┌──────────┐          │  ┌──────────────────┐                       │
   │  图像     │ ────────→ │  YOLO11 (best.pt) │ ──→ [B, 512]          │
   │  RGB     │          │  │  特征提取 + 池化   │     visual emb       │
   └──────────┘          │  └──────────────────┘                       │
                         │                                              │
   ┌──────────┐          │  ┌──────────────────┐                       │
   │  传感器   │ ────────→ │  PatchTST / 1D-CNN│ ──→ [B, 256]          │
   │  C×T     │          │  │  时序编码器        │     sensor emb       │
   └──────────┘          │  └──────────────────┘                       │
                         │                                              │
   ┌──────────┐          │  ┌──────────────────┐                       │
   │  文本报告 │ ────────→ │  BERT / RoBERTa   │ ──→ [B, 256]          │
   │  中文    │          │  │  语义编码器        │     text emb         │
   └──────────┘          │  └──────────────────┘                       │
                         │         │         │                          │
                         │         └────┬────┘                          │
                         │              ▼                               │
                         │  ┌───────────────────────┐                   │
                         │  │  多模态融合模块        │                   │
                         │  │  • Cross-Attention    │  ← 可配置切换     │
                         │  │  • Gated Fusion       │                   │
                         │  │  • Transformer Fusion │                   │
                         │  └───────────────────────┘                   │
                         │              │                               │
                         │              ▼                               │
                         │  ┌───────────────────────┐                   │
                         │  │  分类头 (MLP)         │                   │
                         │  └───────────────────────┘                   │
                         │              │                               │
                         │              ▼                               │
                         │  fallen tree │ landslide │ road collapse │ stone │
                         └──────────────────────────────────────────────┘
```

### 数据流详解

1. **视觉通道**：图像经过 YOLO11 backbone + neck 提取空间特征图 → Attention Pooling → 投影到 512 维
2. **时序通道**：传感器数据经 Patch Embedding → Transformer Encoder → CLS Token → 投影到 256 维
3. **文本通道**：报告经 BERT Tokenizer → BERT Encoder → [CLS] 向量 → 投影到 256 维
4. **融合层**：三个模态向量送入融合模块，输出统一的融合表征
5. **分类头**：融合表征 → MLP → 4 类灾害概率分布

---

## 项目目录结构

```
geohazard/
│
├── README.md                           # 📖 本文件
├── requirements.txt                    # 🐍 Python 依赖清单
├── train_model.py                      # 🚂 训练多模态入口脚本

│
├── configs/                            # ⚙️ 配置模块
│   ├── default.yaml                    #    全局 YAML 配置（模型/训练/数据参数）
│   └── model_config.py                 #    强类型 Dataclass 配置校验
│
├── data/                               # 📦 数据处理
│   ├── data.yaml                       #    ★ YOLO 数据集描述（类别名、路径）
│   ├── dataset.py                      #    多模态 Dataset：直接扫描 YOLO 目录，无需 labels.csv
│   ├── dataloader.py                   #    多模态 Collate Function + DataLoader 工厂
│   └── transforms.py                   #    数据增强（图像/传感器/文本）
│
├── models/                             # 🧠 模型核心
│   ├── geohazard_model.py              #    ★ 顶层模型：串联所有编码器 + 融合 + 分类头
│   │
│   ├── vision/                         #    👁️ 视觉模态
│   │   └── yolo_encoder.py             #    ★ YOLO11 特征提取器（加载 best.pt）
│   │
│   ├── timeseries/                     #    📈 时序模态
│   │   ├── ts_transformer.py           #       PatchTST Transformer 编码器
│   │   └── sensor_encoder.py           #       传感器编码器（Transformer / 1D-CNN）
│   │
│   ├── text/                           #    📝 文本模态
│   │   └── text_encoder.py             #       BERT / RoBERTa 文本编码器
│   │
│   └── fusion/                         #    🔗 多模态融合
│       ├── cross_attention_fusion.py   #       Cross-Attention 融合（ViLT/METER 范式）
│       ├── gated_fusion.py             #       门控融合（Gated Multimodal Units）
│       └── transformer_fusion.py       #       单流 Transformer 融合（UNITER 范式）
│
├── inference/                          # 🚀 推理部署
│   └── predictor.py                    #    推理入口：classify / detect / full 三种模式 + CLI
│
├── utils/                              # 🛠️ 工具模块
│   ├── metrics.py                      #    评估指标（Accuracy/F1/ROC-AUC/混淆矩阵）
│   └── visualization.py                #    可视化（检测框/注意力热力图/模态贡献图）
│
└── weights/                            # 📁 预训练权重
    └── best.pt                         #    ★ 你的 YOLO11 训练权重（放在这里）
 ── tools/                              # 📁 工具
    └── generate_reports.py                        #    ★ 对yolo的图像生成丰富语义现场报告
```

### 核心文件说明

| 文件 | 功能 | 重要性 |
|------|------|--------|
| `models/vision/yolo_encoder.py` | 加载 `best.pt`，冻结 backbone，提取空间特征 | ⭐⭐⭐ |
| `models/geohazard_model.py` | 顶层模型，串联所有模态 + 融合 + 4 类分类头 | ⭐⭐⭐ |
| `inference/predictor.py` | 推理接口：classify / detect / full 三种模式 | ⭐⭐⭐ |
| `data/dataset.py` | 多模态 Dataset，适配 YOLO 目录结构 | ⭐⭐⭐ |
| `configs/default.yaml` | 所有可调参数集中管理 | ⭐⭐ |
| `data/data.yaml` | YOLO 数据集配置（4 类 + 路径） | ⭐⭐ |
| `train.py` | 完整训练流程（AMP + warmup + 早停） | ⭐⭐ |

---

## 快速开始

### 环境要求

- Python ≥ 3.10
- CUDA ≥ 11.8（GPU 训练推荐，CPU 推理亦可）
- 显存 ≥ 8 GB（推荐 16 GB+）

### 安装

```bash
# 1. 进入项目目录
cd E:\transform\geohazard

# 2. 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate   # Linux/Mac

# 3. 安装依赖
pip install -r requirements.txt

# 4. 将你的 YOLO11 best.pt 放入 weights/ 目录
copy 你训练好的best.pt路径 E:\transform\geohazard\weights\best.pt
```

### 一分钟推理测试

```python
from inference.predictor import GeoHazardPredictor

# 初始化（自动加载 YOLO best.pt + 融合模型）
predictor = GeoHazardPredictor(
    config_path="configs/default.yaml",
    checkpoint_path="outputs/best_model.pt"  # 训练后可替换
)

# YOLO 目标检测
detections = predictor.detect("data/raw/test/images/img_001.jpg")
for d in detections:
    print(f"[{d['class_name']}] conf={d['confidence']:.2f} bbox={d['bbox']}")

# 多模态分类（图像 + 传感器 + 文本）
result = predictor.predict(
    image_path="data/raw/test/images/img_001.jpg",
    sensor_path="data/raw/test/sensors/img_001.npy",
    report_text="连续暴雨3天，山坡出现横向裂缝约5厘米宽，有滑动迹象。"
)
print(f"灾害类型: {result['class_name']}")
print(f"置信度:   {result['confidence']:.2%}")

# 综合预测（检测 + 分类一次性完成）
full = predictor.predict_full(
    image_path="data/raw/test/images/img_001.jpg",
    sensor_path="data/raw/test/sensors/img_001.npy",
)
print(f"检测到 {len(full['detections'])} 个目标")
print(f"场景分类: {full['classification']['class_name']}")
```

输出示例：

```
[landslide] conf=0.93 bbox=[120, 85, 540, 380]
[stone]     conf=0.87 bbox=[350, 200, 410, 280]
灾害类型: landslide
置信度:   94.7%
```

### 命令行推理

```bash
# 多模态分类
python -m inference.predictor --mode classify \
    --image ./data/raw/test/images/img_001.jpg \
    --sensor ./data/raw/test/sensors/img_001.npy \
    --report "现场调查报告内容..."

# YOLO 目标检测
python -m inference.predictor --mode detect \
    --image ./data/raw/test/images/img_001.jpg --conf 0.3

# 综合预测（检测 + 分类）
python -m inference.predictor --mode full \
    --image ./data/raw/test/images/img_001.jpg
```

---

## 数据准备

### YOLO 数据集目录结构（你的现有数据）

```
data/raw/
├── data.yaml                  # YOLO 数据集描述文件
├── train/                     # 训练集
│   ├── images/                #   原图 (.jpg / .png)
│   │   ├── img_001.jpg
│   │   └── ...
│   └── labels/                #   YOLO 标注 (.txt, 与图像同名)
│       ├── img_001.txt
│       └── ...
        ├── sensors/                   # ★ 传感器时序数据 (.npy / .csv，可选)
│          ├── station_A_001.npy
│          └── ...
        └── reports/                   # ★ 文本报告 (.txt，可选)
            ├── report_001.txt
            └── ...
│
├── valid/                     # 验证集
│   ├── images/
│   └── labels/
    └── sensors/
    └── reports/
│
├── test/                      # 测试集
│   ├── images/
│   └── labels/
    └── sensors/
    └── reports/

```

### `data/data.yaml` — YOLO 数据集配置

```yaml
path: .
train: train/images
val: valid/images
test: test/images

nc: 4
names:
  0: fallen tree
  1: landslide
  2: road collapse
  3: stone
```

### 无需 labels.csv — 自动推断

系统**直接扫描 YOLO 目录**，自动发现所有图像和标注文件。图像级分类标签 = 标注中出现最多的类别（多数投票）。

传感器和文本报告通过**文件名匹配**自动关联——只需文件名与图像相同、扩展名不同、放在对应目录即可。完全零配置。

### YOLO 标注格式

每张图像对应一个同名的 `.txt` 文件，每行一个目标：

```
class_id x_center y_center width height
```

所有坐标为归一化值（相对于图像宽高，范围 0~1）。

示例 (`img_001.txt`)：
```
1 0.523 0.418 0.312 0.245
3 0.780 0.350 0.080 0.095
```
表示：1 个 landslide (1) + 1 个 stone (3)。

### 传感器数据格式

- **`.npy` 文件**：shape 为 `[channels, time_steps]`，如 `[6, 256]` 表示 6 通道 × 256 时间步
- **`.csv` 文件**：列为通道，行为时间步（自动转置）

---

## 模型配置

所有可调参数集中在 `configs/default.yaml`，核心配置项：

```yaml
# ========== 全局 ==========
num_classes: 4                     # 4 类灾害
class_names:
  - "fallen tree"
  - "landslide"
  - "road collapse"
  - "stone"

# ========== 视觉模态 — 你的 YOLO11 best.pt ==========
vision:
  custom_weights: "weights/best.pt"    # ← 指向你的 best.pt
  data_yaml: "data/data.yaml"         # ← YOLO 数据集描述
  freeze_backbone: true               # 冻结 YOLO（仅训练融合层）

# ========== 数据路径 ==========
data:
  root: "./data/raw"                  # YOLO 数据集根目录
  sensor_dir: "sensors"               # 传感器子目录（可选）
  report_dir: "reports"               # 报告子目录（可选）

# ========== 融合策略（四选一）==========
fusion:
  method: "concat"                    # concat | gated | cross_attention | transformer
```

---

## 训练流程

### 启动训练

```bash
python train_model.py --config configs/default.yaml
```

### 训练过程详解

```
Epoch 1 / 100
==================================================
Train Epoch 1: 100%|████████| loss: 1.0852, lr: 8.2e-05
Train Loss: 1.0234 | Acc: 0.7456 | F1: 0.7123
Val Loss:   0.8912 | Acc: 0.8321 | F1: 0.8102 | ROC-AUC: 0.8956
✓ 保存最优模型 → outputs/best_model.pt (F1=0.8102)
```

### 训练策略

1. **分阶段训练**：YOLO backbone 冻结 → 仅训练融合层 + 分类头
2. **差异学习率**：Vision 模块使用 0.1× 基础学习率
3. **学习率调度**：Linear Warmup（前 1000 步）→ Cosine Annealing
4. **混合精度**：FP16 AMP 训练，节省 ~40% 显存
5. **早停机制**：验证集 F1-macro 连续 10 轮不提升即停止
6. **标签平滑**：CrossEntropyLoss label_smoothing=0.1

### 恢复训练

```bash
python train_model.py --config configs/default.yaml --resume outputs/best_model.pt
```

---

## 推理部署

### Python API — 三种模式

```python
from inference.predictor import GeoHazardPredictor

predictor = GeoHazardPredictor(
    config_path="configs/default.yaml",
    checkpoint_path="outputs/best_model.pt",
)

# 模式 1：多模态分类（image-level）
result = predictor.predict(
    image_path="data/raw/test/images/img_001.jpg",
    sensor_path="data/raw/test/sensors/img_001.npy",
    report_text="现场报告...",
)
# → {"class_name": "landslide", "confidence": 0.947, "probs": {...}}

# 模式 2：YOLO 目标检测（bbox-level）
detections = predictor.detect("data/raw/test/images/img_001.jpg", conf_threshold=0.3)
# → [{"class_name": "landslide", "confidence": 0.93, "bbox": [120,85,540,380]}, ...]

# 模式 3：综合预测（检测 + 分类）
full = predictor.predict_full(
    image_path="data/raw/test/images/img_001.jpg",
    sensor_path="data/raw/test/sensors/img_001.npy",
)
# → {"detections": [...], "classification": {...}}

# 特征提取（用于聚类、检索等下游任务）
embedding = predictor.extract_features(image_path="img.jpg")
# → np.ndarray shape (1, 512)
```

### 缺失模态推理

系统自动处理模态缺失：

```python
# 仅有图像
result = predictor.predict(image_path="test.jpg")
# ✅ 纯视觉分类

# 图像 + 文本（无传感器）
result = predictor.predict(image_path="test.jpg", report_text="裂缝约3cm...")
# ✅ 视觉 + 文本双模态融合
```

---

## 融合策略详解

| 策略 | 配置值 | 核心思想 | 适用场景 |
|------|--------|---------|---------|
| **Concat** (Baseline) | `concat` | 直接拼接 + 线性投影 | 快速原型验证 |
| **Gated Fusion** | `gated` | 门控网络动态学习各模态权重 | 模态经常缺失或噪声大 |
| **Cross-Attention** 🌟 | `cross_attention` | 传感器 Query → 视觉特征 Key/Value | 传感器为主要预警信号（推荐） |
| **Transformer Fusion** | `transformer` | 单流联合 Self-Attention | 三模态质量均衡，数据量充足 |

### 学术参考

| 模块 | 参考 |
|------|------|
| YOLO11 | Jocher, G., et al. "Ultralytics YOLO." 2024. |
| PatchTST (时序) | Nie, Y., et al. "A Time Series is Worth 64 Words." *ICLR 2023.* |
| BERT (文本) | Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers." *NAACL 2019.* |
| Cross-Attention | ViLT (Kim et al., *ICML 2021*) + METER (Dou et al., *CVPR 2022*) |
| Gated Fusion | Arevalo, J., et al. "Gated Multimodal Units." *ICLR-W 2017.* |
| Transformer Fusion | VisualBERT (Li et al., 2019) + UNITER (Chen et al., *ECCV 2020*) |

---

## 常见问题

<details>
<summary><b>Q: best.pt 放在哪里？</b></summary>

放在 `E:\transform\geohazard\weights\best.pt`。配置文件 `configs/default.yaml` 中已预设：

```yaml
vision:
  custom_weights: "weights/best.pt"
```

系统通过 `ultralytics.YOLO("weights/best.pt")` 自动加载。
</details>

<details>
<summary><b>Q: 我的 YOLO 数据集已经分好了 train/valid/test，还需要做什么？</b></summary>

**什么都不需要做。** 系统直接扫描 `data/raw/{train,valid,test}/images/` 发现所有图像，从对应 `labels/` 目录读取标注，自动推断图像级标签。放好数据即可训练。
</details>

<details>
<summary><b>Q: 如果没有传感器和文本数据怎么办？</b></summary>

默认配置已关闭（`enabled: false`），系统自动退化为纯视觉分类+检测。后续有了传感器/文本数据，只需放到对应目录并开启即可。
</details>

<details>
<summary><b>Q: 类别名和数量可以自定义吗？</b></summary>

可以。修改三处保持同步：
1. `data/data.yaml` — `nc` 和 `names`
2. `configs/default.yaml` — `num_classes` 和 `class_names`
3. `configs/model_config.py` — `num_classes` 和 `class_names` 默认值
</details>

<details>
<summary><b>Q: 显存不够怎么办？</b></summary>

- 减小 `batch_size`（如 8 或 4）
- 减小 `timeseries.seq_length`（如 128）
- 减小 `fusion.hidden_dim`（如 256）
- 确保 `mixed_precision: "fp16"` 已开启
</details>

<details>
<summary><b>Q: 推理速度慢，如何优化？</b></summary>

- 时序编码器使用 CNN 模式（在 `sensor_encoder.py` 设置 `backbone="cnn"`）
- 使用 TensorRT 或 ONNX 导出 YOLO 模型
- 减少 `fusion.num_layers`（如 2 层）
</details>

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源。

---

<p align="center">
  <b>🌍 GeoHazard-Multimodal</b><br>
  <i>用多模态 AI 守护地质安全</i><br>
  <sub>4 类灾害 · YOLO11 检测 · Cross-Attention 融合</sub><br><br>
  <sub>Built with PyTorch · Ultralytics YOLO11 · HuggingFace Transformers</sub>
</p>
