# TT100K-YOLO 开发进度

## 2026-06-01

### ✅ Task 1: 项目骨架搭建
- 目录结构、requirements.txt、.gitignore、README.md
- 6 个实验 YAML 配置 (exp1 ~ exp4)
- Conda 环境 tt100k: PyTorch 2.6+cu124, Ultralytics 8.4
- 已推送: `9c121cf` → origin/master

### ✅ Task 2: 类别分布统计
- `src/preprocess/class_stats.py` — 适配 TT100K 2021 JSON 真实格式
- 实际运行结果: 10,592 图 / 27,346 目标 / 201 类有实例
- 45 类 ≥100 实例, 小目标占 43.6%
- 已推送: `ecc1f0a`

### ✅ Task 3: 类别筛选
- `src/preprocess/filter.py` — 阈值 ≥100 → 45 类
- 保留 24,212 对象, 过滤 3,134, 剩余 9,738 张图
- 已推送: `7fa91f9`

### ✅ Task 4: 格式转换
- `src/preprocess/convert.py` — JSON → YOLO txt (归一化坐标)
- 9738 个 txt 文件, 零漏转
- 已推送: `7fa91f9`

### ✅ Task 5: 数据集划分
- `src/preprocess/split.py` — 分层 7:2:1
- train: 6,816 / val: 1,948 / test: 974
- 已推送: `16decb4`

### ⬜ Task 6: Per-class AP 回调
### ⬜ Task 7: 实验配置文件 (已完成YAML，待验证)
### ⬜ Task 8: 实验运行器
### ⬜ Task 9: 绘图脚本
### ⬜ Task 10: Flask 推理 API
### ⬜ Task 11: 数据下载辅助脚本
### ⬜ Task 12: 集成验证

---

## 数据流水线产出

```
data/
├── raw/tt100k_2021/           ← 原始数据 (10,592 图, annotations_all.json)
├── processed/
│   ├── stats/                 ← Task 2: class_distribution.csv, class_summary.txt
│   ├── filtered/              ← Task 3: annotations_filtered.json (9,738 图)
│   └── yolo/                  ← Task 4+5
│       ├── labels/            ← 9738 txt (YOLO format)
│       ├── images/train/      ← 6816 jpg
│       ├── images/val/        ← 1948 jpg
│       ├── images/test/       ← 974 jpg
│       └── dataset.yaml       ← YOLO 训练配置 (45 类)
```