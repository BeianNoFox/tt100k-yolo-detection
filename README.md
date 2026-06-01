# TT100K 交通标志检测系统

本科毕设项目，基于 YOLOv8 和 TT100K 数据集的中国交通标志检测。

## 目录

- `src/preprocess/` — 数据预处理（统计、筛选、格式转换、划分、切片）
- `src/train/` — 训练脚本与回调
- `src/eval/` — 评估与绘图
- `src/infer/` — Flask 推理 API
- `configs/` — 实验 YAML 配置
- `experiments/` — 实验输出（日志、权重、图表）

## 环境

Python 3.10+, PyTorch 2.x, CUDA 12.4+

安装：`pip install -r requirements.txt`
