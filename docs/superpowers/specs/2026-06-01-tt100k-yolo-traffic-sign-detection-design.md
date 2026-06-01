# TT100K 交通标志检测系统 — 设计规格

## 项目概述

本科毕设，人工智能专业，工程导向（训练模型 + 论文 + Web Demo）。基于 TT100K 2021 数据集和 YOLOv8 进行中国交通标志的目标检测，重点解决小目标检测和长尾分类问题。

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 检测框架 | Ultralytics YOLOv8s | 社区活跃、文档完善、内置增强/Loss/Anchor，工程效率高 |
| 数据集 | TT100K 2021 (232类→筛选~46类) | 中国交通标志最大规模数据集，2048×2048 街景 |
| DL 框架 | PyTorch | YOLOv8 底层依赖 |
| 推理服务 | Flask + ONNX/PT | 轻量，易部署演示 |
| 展示层 | HTML/CSS/JS (后续单独设计) | 不纳入本轮规格 |

## 系统架构

```
┌──────────────────────────────────┐
│        推理服务层 (Flask API)     │
│   模型加载 / 大图切片推理 / 后处理 │
└──────────────┬───────────────────┘
               │ 模型文件 (.pt/.onnx)
┌──────────────┴───────────────────┐
│         数据与训练层 (离线)        │
│ 预处理 → 类筛选 → 切片 → 训练 → 评估 │
└──────────────────────────────────┘
```

展示层后续单独设计，本轮只涉及数据与训练层 + 推理服务层。

---

## 模块一：数据预处理

### 1.1 类别分布统计与尺寸分析

**输入**：TT100K 2021 原始 JSON 标注

**产出**：
- 全量 232 类实例数分布表/图 → 支撑筛选阈值选取（论文 2.1.1 节）
- bbox 尺寸分布统计（原始 2048px 下像素尺寸，分小/中/大桶）→ 支撑切片动机（论文 2.1.2 节）

**关键发现（已知）**：
- 70+ 类为空类，~162 类实例 <100
- 平均 bbox 高度 ~42-55px，在 2048px 图中占比仅 0.04%-0.10%

### 1.2 类别筛选

- 阈值：实例数 ≥ 50
- 产出：约 46 类
- 输出：筛选后的标注文件 + 类别映射表（论文附录）
- 筛选后对长尾仍需处理（最多类 3000+ vs 最少类 50，差 60 倍）

### 1.3 标注格式转换

- JSON → YOLO txt 格式
- 每图一个 .txt，每行：`class_id cx cy w h`（归一化到 [0,1]）
- class_id 重新映射为 0-45 的连续编号

### 1.4 数据集划分

- 训练集 : 验证集 : 测试集 = 7:2:1
- 分层采样，确保各类比例一致
- 不沿用 TT100K 自带划分（train/test/other 不完全标准）
- TT100K 的 "Other" 部分（7641 张）无标注，直接丢弃不用

---

## 模块二：消融实验设计

四个实验递进，每个在前一个基础上加一项改进。

### Exp 1 — 基线

| 配置 | 值 |
|------|----|
| 模型 | YOLOv8s |
| 输入尺寸 | 640×640（直接 resize 原图） |
| Epochs | 200, Patience=50 |
| 预训练 | COCO backbone |
| 增强 | 默认（仅基础翻转/hsv） |
| 长尾处理 | 无 |

**目的**：取得基准 mAP，同时运行类质量审查。

### 类质量审查（Exp 1 完成后）

- 对 Per-class AP < 5% 的类，考察其样本质量（标注噪声、样本多样性）
- 决定策略：删除该类 或 合并到语义相近类
- 审查后确定最终类数（可能小于 46），更新类别映射

### Exp 2 — 大图切片

| 配置 | 值 |
|------|----|
| 切片尺寸 | 640×640 |
| Overlap 对比 | 0% / 20% / 30%（三组内部跑，选最优与 Exp1 比） |
| 其他 | 与 Exp 1 相同 |

**切片流程**：2048×2048 → 640×640 子图（带 overlap），重算每块内 bbox 坐标。推理时拼接子图结果，NMS 合并跨块框。

**目的**：验证切片对目标检测的改善，overlap 对比选最优策略。

### Exp 3 — 长尾处理

| 配置 | 值 |
|------|----|
| 加倍采样 | 对实例 <200 的类，按档位复制 3x-10x |
| 类别权重 Loss | 低频类更高权重 |
| Mosaic/MixUp | 辅助增强 |
| 其他 | 继承 Exp 2 最优切片配置 |

**优先级**：加倍采样 > 类别权重 > Mosaic（避免头部欠拟合）

**目的**：验证长尾处理对中低频类 AP 的改善。

### Exp 4 — 最终调参

| 配置 | 值 |
|------|----|
| 优化器 | AdamW |
| 初始 LR | 调优（对比 0.001/0.0005/0.0001） |
| Batch Size | 按 GPU 显存拉满 |
| 多尺度训练 | 开（0.5x-1.5x） |
| 其他 | 继承 Exp 3 全部配置 |

**目的**：冲最高 mAP。

---

## 模块三：实验日志与训练日记

每个实验的训练过程必须产出 **完整可复现的日志**，训练结束后可通过日志脚本重绘论文所需全部图表，无需重新训练。

### 3.1 每个实验的输出目录结构

```
experiments/{exp_name}/
├── config.yaml              # 本次实验的完整训练参数（原始副本）
├── args.yaml                # Ultralytics 自动保存的训练参数
├── results.csv              # 每 epoch 全部指标（Ultralytics 自动产出）
├── per_class_ap.csv         # 每 epoch 各类 AP（自定义 callback 采集）
├── best.pt                  # 最优权重
├── last.pt                  # 最后一轮权重
├── model_summary.txt        # 模型结构摘要
├── plots/                   # 训练过程图（部分 Ultralytics 自动 + 自定义）
│   ├── results.png          # 综合指标面板（Ultralytics 自动）
│   ├── loss_curves.png      # train/val loss 分离折线图（自定义）
│   ├── map_curve.png        # mAP@0.5 / mAP@0.5:0.95 随 epoch 变化（自定义）
│   ├── lr_schedule.png      # 学习率变化曲线（自定义）
│   └── precision_recall.png # P/R 随 epoch 变化（自定义）
├── eval/                    # 验证集评测结果
│   ├── confusion_matrix.png
│   ├── PR_curve.png
│   ├── F1_curve.png
│   ├── P_curve.png
│   ├── R_curve.png
│   ├── per_class_ap_bar.png # 每类 AP 柱状图（自定义）
│   └── per_class_ap.csv     # 最终每类 AP 表
└── predictions/             # 测试集预测样本
    ├── correct/             # 正确检测样例
    ├── false_positive/      # 误检样例
    └── false_negative/      # 漏检样例
```

### 3.2 实验配置文件模板

每个实验的 `config.yaml` 包含以下字段，按实验号覆写不同值：

```yaml
experiment: "exp1_baseline"
model: "yolov8s.pt"
imgsz: 640
epochs: 200
patience: 50
batch: 16          # 按 GPU 显存调整
lr0: 0.001         # 初始学习率
optimizer: "AdamW"
pretrained: true
# --- 切片配置 ---
tiling:
  enabled: false
  tile_size: 640
  overlap: 0.0
# --- 增强 ---
augment:
  mosaic: 0.0
  mixup: 0.0
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  flipud: 0.0
  fliplr: 0.5
# --- 长尾处理 ---
long_tail:
  oversample: false      # 加倍采样开关
  oversample_threshold: 200
  oversample_multiplier: [3, 5, 10]  # 分档倍率
  class_weight_loss: false
```

### 3.3 日志采集机制

**Ultralytics 自动产出**（无需额外代码）：
- `args.yaml` — 训练参数快照
- `results.csv` — 每 epoch 的 train/val box_loss, cls_loss, dfl_loss, precision, recall, mAP@0.5, mAP@0.5:0.95
- `results.png` — 综合曲线面板
- `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png`, `P_curve.png`, `R_curve.png`
- `train_batch*.jpg`, `val_batch_pred.jpg` — 训练/验证 batch 预览

**自定义 Callback 补充采集**（需要实现）：
- `per_class_ap.csv`：Ultralytics 每 epoch 不输出 per-class AP，需要写一个 callback 在 `on_val_end` 钩子中计算并追加写入
- `model_summary.txt`：训练开始前通过 `torchsummary` 或 Ultralytics `model.info()` 输出模型参数量、FLOPs、层数

### 3.4 自定义绘图脚本（`src/eval/plot_metrics.py`）

训练完成后运行，读取 `results.csv` 和 `per_class_ap.csv`，用 matplotlib 生成高质量 300dpi 图表，统一中文字体、配色、字号，满足论文直接使用。

**产出图表清单**：

| 图表 | 数据源 | 论文位置 |
|------|--------|---------|
| Loss 训练/验证分轨折线图（box+cls+dfl 三合一） | results.csv | 实验结果章节 |
| mAP@0.5 & mAP@0.5:0.95 双轴折线图 | results.csv | 实验结果章节 |
| Precision/Recall 双轴折线图 | results.csv | 实验结果章节 |
| 学习率衰减曲线 | results.csv (lr 列) | 训练细节节 |
| 各实验 Loss 下降对比（多实验叠在一张图） | 各实验 results.csv | 消融分析节 |
| 各实验 mAP 提升对比柱状图 | 各实验 best mAP | 消融分析节 |
| 各实验 × 各类 AP 热力图 | 各实验 per_class_ap.csv | 消融分析节 |
| Confusion Matrix 热力图 | Ultralytics 自动 | 类别分析节 |
| Per-class AP 降序柱状图 | per_class_ap.csv | 类别分析节 |
| 小/中/大目标分桶 mAP 对比柱状图 | 自定义评测脚本 | 小目标分析节 |
| 类别实例数 vs AP 散点图（揭示长尾影响） | 类别统计 + AP | 长尾分析节 |

### 3.5 消融对比表（论文核心论据）

| 实验 | mAP@0.5 | mAP@0.5:0.95 | 说明 |
|------|---------|--------------|------|
| Exp1 基线 | TBD | TBD | 直接 resize，无优化 |
| Exp2 +切片 | TBD | TBD | 相对 Exp1 提升 Δ |
| Exp3 +长尾处理 | TBD | TBD | 相对 Exp2 提升 Δ |
| Exp4 最终调参 | TBD | TBD | 相对 Exp3 提升 Δ |

### 3.6 Per-class 分析

- 高/中/低频类分桶 AP 统计表
- 识别最差 5 类，分析原因（样本量？标注噪声？类间相似？）

### 3.7 小目标分桶评估

- bbox 面积 <32^2, 32^2-96^2, >96^2 三档
- 突出切片策略对各档的提升幅度

### 3.8 检测结果可视化

- 测试集抽样：正确检测、误检（FP）、漏检（FN）各存若干张
- 框图标注格式统一（类别名+置信度，中文标注）

---

## 模块四：推理服务层

### 4.1 模型加载

- 加载 Exp 4 产出的 best.pt
- 可选：导出 ONNX 用于加速

### 4.2 推理逻辑

- 输入图片 → 大图切片（按最优 overlap）→ 每块送模型 → NMS 合并
- 输出 JSON：`[{bbox, class_name, confidence}]`

### 4.3 Flask API

```
POST /detect
  Content-Type: multipart/form-data (image file)
  Response: JSON [{class_name, confidence, x1, y1, x2, y2}]

POST /detect_batch
  Content-Type: multipart/form-data (multiple image files)
  Response: JSON per image
```

---

## 目标指标预估（46 类场景）

| 指标 | 预估 |
|------|------|
| mAP@0.5 (Exp 4 最终) | 65-75% |
| 高频类 AP (样本>500) | 80-90% |
| 中频类 AP (样本 100-500) | 55-75% |
| 低频类 AP (样本 50-100) | 30-50% |

---

## 项目结构

```
tabel/
├── data/                    # 数据目录（gitignore）
│   ├── raw/                 # TT100K 原始数据
│   └── processed/           # 预处理后数据
├── configs/                 # 训练配置文件
├── src/
│   ├── preprocess/          # 数据预处理脚本
│   │   ├── class_stats.py   # 类别分布统计
│   │   ├── filter.py        # 类别筛选
│   │   ├── convert.py       # 标注格式转换
│   │   ├── split.py         # 数据集划分
│   │   └── tiling.py        # 大图切片
│   ├── train/               # 训练脚本
│   ├── eval/                # 评估分析脚本
│   │   └── plot_metrics.py  # 从日志生成论文图表
│   ├── infer/               # 推理服务（Flask）
│   │   ├── app.py            # Flask 主入口 + API
│   │   └── templates/        # 前端页面（后续单独设计）
│   └── utils/               # 通用工具
├── experiments/             # 实验记录（YAML 配置 + 结果日志）
├── docs/                    # 文档
│   └── superpowers/specs/   # 设计规格（本文档）
└── README.md
```

---

## 实验环境

| 项目 | 要求 |
|------|------|
| Python | 3.10+ |
| PyTorch | 2.x |
| Ultralytics | 8.x |
| GPU | 建议 >= 8GB VRAM（RTX 3060+），最低 4GB |
| CUDA | 11.8+ |
