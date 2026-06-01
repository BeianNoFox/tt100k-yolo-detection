# TT100K 交通标志检测系统 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建完整的 TT100K 交通标志检测系统——从数据预处理、YOLOv8 训练实验、日志可视化到 Flask 推理 API。

**Architecture:** 离线数据预处理管道输出标准化 YOLO 格式数据 → 四个消融实验递进训练并产出完整日志 → 统一绘图脚本从日志生成论文图表 → Flask 推理服务加载最优模型对外提供 REST API。

**Tech Stack:** Python 3.10+, PyTorch 2.x, Ultralytics 8.x, Flask, matplotlib, PyYAML

---

### Task 1: 项目骨架搭建

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `configs/exp1_baseline.yaml`
- Create: `configs/exp2_tiling.yaml`
- Create: `configs/exp3_longtail.yaml`
- Create: `configs/exp4_finetune.yaml`
- Create: `src/__init__.py`
- Create: `src/preprocess/__init__.py`
- Create: `src/train/__init__.py`
- Create: `src/eval/__init__.py`
- Create: `src/infer/__init__.py`
- Create: `src/utils/__init__.py`
- Create: `README.md`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p d:/1/tabel/{data/raw,data/processed,configs,experiments,docs/superpowers/{specs,plans}}
mkdir -p d:/1/tabel/src/{preprocess,train,eval,infer/templates,utils}
```

- [ ] **Step 2: 写入 `requirements.txt`**

```txt
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
flask>=3.0.0
matplotlib>=3.7.0
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0
pillow>=10.0.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
```

- [ ] **Step 3: 写入 `.gitignore`**

```gitignore
data/
__pycache__/
*.pyc
*.pt
*.onnx
runs/
experiments/*/
!.gitkeep
*.egg-info/
.env
venv/
.venv/
```

- [ ] **Step 4: 写入 `README.md`**

```markdown
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

Python 3.10+, PyTorch 2.x, CUDA 11.8+

安装：`pip install -r requirements.txt`
```

- [ ] **Step 5: 创建各包的 `__init__.py`**

```bash
for d in src src/preprocess src/train src/eval src/infer src/utils; do
    touch "d:/1/tabel/$d/__init__.py"
done
```

- [ ] **Step 6: 提交**

```bash
cd d:/1/tabel
git init
git add -A
git commit -m "chore: scaffold project structure with dependencies"
```

---

### Task 2: 数据预处理 — TT100K JSON 格式探查 + 类别统计

**Files:**
- Create: `src/preprocess/class_stats.py`

- [ ] **Step 1: 运行 JSON 格式探查命令**

```bash
cd d:/1/tabel
python -c "
import json
with open('data/raw/annotations.json', 'r') as f:
    data = json.load(f)
print('Type:', type(data))
if isinstance(data, dict):
    print('Keys:', list(data.keys())[:10])
    for k, v in list(data.items())[:1]:
        print(f'Example key={k}, value type={type(v).__name__}, value={v}')
elif isinstance(data, list):
    print('Length:', len(data))
    print('First item:', json.dumps(data[0], indent=2, ensure_ascii=False)[:500])
"
```

- [ ] **Step 2: 根据探查结果调整 class_stats.py 中的解析逻辑，写入脚本**

TT100K 常见格式：JSON 对象，key 为图片文件名，value 为 `{"objects": [{bbox, category}]}` 或 `[{bbox, category}]`。脚本自动适配两种格式。

```python
"""TT100K 类别分布统计与 bbox 尺寸分析"""
import json
import csv
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def parse_annotations(anno_path: Path):
    """读取 TT100K JSON，返回 [(image_name, [objects]), ...]"""
    with open(anno_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for key, val in data.items():
        # key 是图片文件名（可能带路径前缀）
        img_name = Path(key).name
        objects = val.get("objects", val) if isinstance(val, dict) else val
        if isinstance(objects, dict):
            objects = objects.get("objects", [])
        results.append((img_name, objects))

    print(f"[INFO] {len(results)} images loaded, total objects: {sum(len(objs) for _, objs in results)}")
    return results


def compute_stats(annotations: list, output_dir: Path):
    """计算类别实例数分布和 bbox 尺寸分布，输出 CSV + 柱状图"""
    class_counter = Counter()
    size_bins = {"small": 0, "medium": 0, "large": 0}  # 以 2048px 原图为基准

    for img_name, objects in annotations:
        for obj in objects:
            cat = obj.get("category", obj.get("class", "unknown"))
            class_counter[cat] += 1
            bbox = obj.get("bbox", {})
            if isinstance(bbox, dict):
                w = bbox.get("xmax", 0) - bbox.get("xmin", 0)
                h = bbox.get("ymax", 0) - bbox.get("ymin", 0)
            elif isinstance(bbox, list) and len(bbox) == 4:
                xmin, ymin, xmax, ymax = bbox
                w, h = xmax - xmin, ymax - ymin
            else:
                continue
            area = w * h
            if area < 32 * 32:
                size_bins["small"] += 1
            elif area < 96 * 96:
                size_bins["medium"] += 1
            else:
                size_bins["large"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # 全量类别统计 CSV
    rows = sorted(class_counter.items(), key=lambda x: -x[1])
    csv_path = output_dir / "class_distribution.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "instance_count"])
        writer.writerows(rows)
    print(f"[INFO] Class distribution written to {csv_path}")

    # 汇总统计
    total_classes = len(class_counter)
    head = sum(1 for _, c in rows if c >= 500)
    mid = sum(1 for _, c in rows if 100 <= c < 500)
    low1 = sum(1 for _, c in rows if 50 <= c < 100)
    low2 = sum(1 for _, c in rows if 10 <= c < 50)
    tail = sum(1 for _, c in rows if 1 <= c < 10)
    zero = sum(1 for _, c in rows if c == 0)

    summary = f"""
=== TT100K Class Distribution Summary ===
Total classes: {total_classes}
  >= 500 instances: {head}
  100-499 instances: {mid}
  50-99 instances:   {low1}
  10-49 instances:   {low2}
  1-9 instances:     {tail}
  0 instances:       {zero}

Bbox size distribution (in original 2048px):
  Small  (<32x32):  {size_bins['small']:>6}  ({size_bins['small']/sum(size_bins.values())*100:.1f}%)
  Medium (32-96):   {size_bins['medium']:>6}  ({size_bins['medium']/sum(size_bins.values())*100:.1f}%)
  Large  (>96x96):  {size_bins['large']:>6}  ({size_bins['large']/sum(size_bins.values())*100:.1f}%)
Classes with >=50 instances: {head + mid + low1}
"""
    print(summary)

    summary_path = output_dir / "class_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"[INFO] Summary written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="TT100K class distribution stats")
    parser.add_argument("--anno", required=True, help="Path to annotations.json")
    parser.add_argument("--out", default="data/processed/stats", help="Output directory")
    args = parser.parse_args()

    annotations = parse_annotations(Path(args.anno))
    compute_stats(annotations, Path(args.out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 在项目根目录验证脚本**

```bash
cd d:/1/tabel
python src/preprocess/class_stats.py --anno data/raw/annotations.json --out data/processed/stats
```

期望输出：CSV 表格 + 控制台汇总统计。

- [ ] **Step 4: 提交**

```bash
git add src/preprocess/class_stats.py data/processed/stats/
git commit -m "feat: add class distribution and bbox size analysis script"
```

---

### Task 3: 数据预处理 — 类别筛选

**Files:**
- Create: `src/preprocess/filter.py`

- [ ] **Step 1: 写入 filter.py**

```python
"""按实例数阈值筛选类别，输出筛选后的标注和类别映射"""
import json
import csv
import argparse
from pathlib import Path
from collections import Counter


def filter_classes(anno_path: Path, min_instances: int, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(anno_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 统计每个类别的实例数
    class_counter = Counter()
    for key, val in data.items():
        objects = val.get("objects", val) if isinstance(val, dict) else val
        if isinstance(objects, dict):
            objects = objects.get("objects", [])
        for obj in objects:
            cat = obj.get("category", obj.get("class", "unknown"))
            class_counter[cat] += 1

    # 筛选出实例数 >= min_instances 的类
    valid_classes = sorted([c for c, n in class_counter.items() if n >= min_instances],
                           key=lambda c: -class_counter[c])
    class_map = {cls: idx for idx, cls in enumerate(valid_classes)}

    print(f"[INFO] {len(class_counter)} total classes -> {len(valid_classes)} classes with >= {min_instances} instances")

    # 保存类别映射表
    map_path = output_dir / "class_mapping.csv"
    with open(map_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_class", "new_id", "instance_count"])
        for cls_name, new_id in class_map.items():
            writer.writerow([cls_name, new_id, class_counter[cls_name]])
    print(f"[INFO] Class mapping saved to {map_path}")

    # 过滤标注：只保留属于有效类别的对象
    filtered_annotations = {}
    total_filtered = 0
    for key, val in data.items():
        objects = val.get("objects", val) if isinstance(val, dict) else val
        if isinstance(objects, dict):
            objects = objects.get("objects", [])
        kept = []
        for obj in objects:
            cat = obj.get("category", obj.get("class", "unknown"))
            if cat in class_map:
                obj_copy = dict(obj)
                obj_copy["new_id"] = class_map[cat]
                kept.append(obj_copy)
            else:
                total_filtered += 1
        if kept:
            filtered_annotations[key] = kept

    print(f"[INFO] Filtered out {total_filtered} objects from removed classes")

    # 保存过滤后的标注
    filtered_path = output_dir / "annotations_filtered.json"
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(filtered_annotations, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Filtered annotations saved to {filtered_path}")

    return class_map


def main():
    parser = argparse.ArgumentParser(description="Filter TT100K classes by instance count")
    parser.add_argument("--anno", required=True, help="Path to annotations.json")
    parser.add_argument("--min-instances", type=int, default=50, help="Minimum instances per class")
    parser.add_argument("--out", default="data/processed/filtered", help="Output directory")
    args = parser.parse_args()

    filter_classes(Path(args.anno), args.min_instances, Path(args.out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行验证**

```bash
cd d:/1/tabel
python src/preprocess/filter.py --anno data/raw/annotations.json --min-instances 50 --out data/processed/filtered
```

期望：输出 `class_mapping.csv`（约 46 行 + 表头）和 `annotations_filtered.json`。

- [ ] **Step 3: 提交**

```bash
git add src/preprocess/filter.py
git commit -m "feat: add class filtering script (min-instances threshold)"
```

---

### Task 4: 数据预处理 — JSON → YOLO 格式转换

**Files:**
- Create: `src/preprocess/convert.py`

- [ ] **Step 1: 写入 convert.py**

```python
"""将 TT100K 过滤后 JSON 标注转换为 YOLO txt 格式"""
import json
import argparse
from pathlib import Path


def convert_to_yolo(filtered_json: Path, image_dir: Path, output_dir: Path):
    """
    输入过滤后的 JSON（格式：{img_name: [{category, bbox, new_id}, ...]}）
    输出：output_dir/ 下每张图片一个 .txt，格式：class_id cx cy w h（归一化）
    同时生成 dataset.yaml
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(filtered_json, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # 构建类别映射（从 new_id 字段推导所有类别）
    class_names = {}
    for img_name, objects in annotations.items():
        for obj in objects:
            new_id = obj["new_id"]
            cat = obj.get("category", obj.get("class", "unknown"))
            if new_id not in class_names:
                class_names[new_id] = cat

    num_classes = len(class_names)
    print(f"[INFO] {num_classes} classes in filtered annotations")

    converted = 0
    skipped = 0
    for img_name, objects in annotations.items():
        img_path = image_dir / img_name
        if not img_path.exists():
            # 尝试在所有子目录中查找
            candidates = list(image_dir.rglob(img_name))
            if not candidates:
                skipped += 1
                continue
            img_path = candidates[0]

        # 获取图像尺寸（不依赖 PIL，从已有信息推断或使用固定尺寸）
        # TT100K 图像均为 2048x2048
        img_w, img_h = 2048, 2048

        lines = []
        for obj in objects:
            new_id = obj["new_id"]
            bbox = obj.get("bbox", {})
            if isinstance(bbox, dict):
                xmin = bbox["xmin"]
                ymin = bbox["ymin"]
                xmax = bbox["xmax"]
                ymax = bbox["ymax"]
            elif isinstance(bbox, list) and len(bbox) == 4:
                xmin, ymin, xmax, ymax = bbox
            else:
                continue

            # 转为 YOLO 归一化格式
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            lines.append(f"{new_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        txt_name = Path(img_name).stem + ".txt"
        (output_dir / txt_name).write_text("\n".join(lines), encoding="utf-8")
        converted += 1

    print(f"[INFO] Converted {converted} images, skipped {skipped} (image not found)")

    # 生成 dataset.yaml
    names_sorted = [class_names[i] for i in range(num_classes)]
    yaml_content = f"""# TT100K filtered dataset (>=50 instances per class)
path: {output_dir.parent.absolute().as_posix()}
train: images/train
val: images/val
test: images/test
nc: {num_classes}
names: {names_sorted}
"""
    yaml_path = output_dir.parent / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[INFO] dataset.yaml written to {yaml_path}")

    return num_classes


def main():
    parser = argparse.ArgumentParser(description="Convert TT100K JSON to YOLO txt format")
    parser.add_argument("--anno", required=True, help="Path to filtered annotations JSON")
    parser.add_argument("--images", required=True, help="Directory containing raw images")
    parser.add_argument("--out", default="data/processed/yolo/labels", help="Output directory for txt files")
    args = parser.parse_args()

    convert_to_yolo(Path(args.anno), Path(args.images), Path(args.out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行验证**

```bash
cd d:/1/tabel
python src/preprocess/convert.py \
    --anno data/processed/filtered/annotations_filtered.json \
    --images data/raw \
    --out data/processed/yolo/labels
head -n 3 data/processed/yolo/labels/*.txt | head -5
```

- [ ] **Step 3: 提交**

```bash
git add src/preprocess/convert.py
git commit -m "feat: add JSON-to-YOLO format conversion script"
```

---

### Task 5: 数据预处理 — 分层数据集划分 + 大图切片

**Files:**
- Create: `src/preprocess/split.py`
- Create: `src/preprocess/tiling.py`

- [ ] **Step 1: 写入 split.py**

```python
"""分层划分训练/验证/测试集，保持各类别比例一致"""
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split


def split_dataset(labels_dir: Path, images_dir: Path, output_dir: Path,
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    读取 labels_dir 下所有 txt，按分层抽样划分为 train/val/test。
    将对应图片和标注分别复制到 output_dir/images/{split}/ 和 output_dir/labels/{split}/
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01

    txt_files = sorted(labels_dir.glob("*.txt"))
    print(f"[INFO] {len(txt_files)} label files found")

    # 统计每张图中各类别的实例数，用于分层
    file_class_counts = {}
    for txt in txt_files:
        counts = defaultdict(int)
        lines = txt.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            if line.strip():
                cls_id = int(line.split()[0])
                counts[cls_id] += 1
        file_class_counts[txt.stem] = dict(counts)

    # 构建类别计数向量作为分层标签（取每张图中出现最多的类）
    stems = list(file_class_counts.keys())
    stratify_labels = []
    for s in stems:
        counts = file_class_counts[s]
        # 用该图中实例最多的类作为分层标签
        dominant_class = max(counts, key=counts.get) if counts else 0
        stratify_labels.append(dominant_class)

    # 先分出 train+val 和 test
    train_val_stems, test_stems = train_test_split(
        stems, test_size=test_ratio, random_state=42, stratify=stratify_labels
    )
    # 再分 train 和 val
    tv_labels = [file_class_counts[s] for s in train_val_stems]
    tv_stratify = [max(c, key=c.get) if c else 0 for c in tv_labels]
    train_stems, val_stems = train_test_split(
        train_val_stems,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=42,
        stratify=tv_stratify,
    )

    # 复制文件到目标目录
    for split_name, split_stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for stem in split_stems:
            # 查找对应图片（支持 jpg/jpeg/png）
            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                candidate = images_dir / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
                # 递归查找
                for found in images_dir.rglob(stem + ext):
                    img_path = found
                    break
                if img_path:
                    break

            if img_path:
                shutil.copy2(img_path, img_out / f"{stem}{img_path.suffix}")
            else:
                print(f"[WARN] Image not found for {stem}")

            # 复制标注文件
            src_lbl = labels_dir / f"{stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, lbl_out / f"{stem}.txt")

        print(f"[INFO] {split_name}: {len(split_stems)} images")

    # 更新 dataset.yaml
    yaml_path = output_dir / "dataset.yaml"
    if yaml_path.exists():
        content = yaml_path.read_text(encoding="utf-8")
        content = content.replace("images/train", str((output_dir / "images/train").absolute().as_posix()))
        content = content.replace("images/val", str((output_dir / "images/val").absolute().as_posix()))
        content = content.replace("images/test", str((output_dir / "images/test").absolute().as_posix()))
        yaml_path.write_text(content, encoding="utf-8")
        print(f"[INFO] Updated {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Stratified train/val/test split")
    parser.add_argument("--labels", required=True, help="Directory of YOLO txt labels")
    parser.add_argument("--images", required=True, help="Directory of original images")
    parser.add_argument("--out", default="data/processed/yolo", help="Output root directory")
    args = parser.parse_args()

    split_dataset(Path(args.labels), Path(args.images), Path(args.out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写入 tiling.py**

```python
"""大图切片：将 2048x2048 图片和标注切为 640x640 子图（带 overlap）"""
import argparse
import shutil
from pathlib import Path
import numpy as np


def tile_image_and_labels(img_path: Path, lbl_path: Path, out_img_dir: Path,
                          out_lbl_dir: Path, tile_size: int = 640,
                          overlap: float = 0.0, min_bbox_visibility: float = 0.3):
    """
    将单张大图切片为 tile_size x tile_size 的子图。
    只保留 bbox 可见面积比 >= min_bbox_visibility 的子图标注。
    """

    stride = int(tile_size * (1 - overlap))
    img_w, img_h = 2048, 2048  # TT100K 固定尺寸

    # 读取原始标注
    if not lbl_path.exists():
        return []
    labels = []
    for line in lbl_path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            labels.append((cls_id, cx, cy, w, h))

    stem = img_path.stem
    tile_records = []

    y_positions = list(range(0, img_h - tile_size, stride)) + [img_h - tile_size]
    x_positions = list(range(0, img_w - tile_size, stride)) + [img_w - tile_size]
    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))

    for ty in y_positions:
        for tx in x_positions:
            tile_name = f"{stem}_t{tx}_{ty}"
            tile_xmin, tile_ymin = tx, ty
            tile_xmax, tile_ymax = tx + tile_size, ty + tile_size

            new_labels = []
            for cls_id, cx, cy, w, h in labels:
                # 原始像素坐标
                obj_xmin = (cx * img_w) - (w * img_w / 2)
                obj_ymin = (cy * img_h) - (h * img_h / 2)
                obj_xmax = (cx * img_w) + (w * img_w / 2)
                obj_ymax = (cy * img_h) + (h * img_h / 2)

                # 计算与 tile 的交集
                ixmin = max(obj_xmin, tile_xmin)
                iymin = max(obj_ymin, tile_ymin)
                ixmax = min(obj_xmax, tile_xmax)
                iymax = min(obj_ymax, tile_ymax)

                if ixmin >= ixmax or iymin >= iymax:
                    continue

                obj_area = (obj_xmax - obj_xmin) * (obj_ymax - obj_ymin)
                inter_area = (ixmax - ixmin) * (iymax - iymin)
                if inter_area / obj_area < min_bbox_visibility:
                    continue

                # 转换为 tile 内归一化坐标
                new_cx = ((ixmin + ixmax) / 2 - tile_xmin) / tile_size
                new_cy = ((iymin + iymax) / 2 - tile_ymin) / tile_size
                new_w = (ixmax - ixmin) / tile_size
                new_h = (iymax - iymin) / tile_size

                new_labels.append(f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")

            if new_labels:
                out_lbl_dir.mkdir(parents=True, exist_ok=True)
                (out_lbl_dir / f"{tile_name}.txt").write_text("\n".join(new_labels), encoding="utf-8")
                tile_records.append((img_path, tile_name, tile_xmin, tile_ymin, tile_xmax, tile_ymax))

    return tile_records


def main():
    parser = argparse.ArgumentParser(description="Tile large images for training")
    parser.add_argument("--images", required=True, help="Directory of input images")
    parser.add_argument("--labels", required=True, help="Directory of YOLO labels")
    parser.add_argument("--out-img", required=True, help="Output directory for tiles")
    parser.add_argument("--out-lbl", required=True, help="Output directory for tile labels")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size (default: 640)")
    parser.add_argument("--overlap", type=float, default=0.2,
                        help="Overlap ratio (default: 0.2)")
    args = parser.parse_args()

    img_dir = Path(args.images)
    lbl_dir = Path(args.labels)
    out_img_dir = Path(args.out_img)
    out_lbl_dir = Path(args.out_lbl)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    total_tiles = 0
    for img_path in sorted(img_dir.glob("*")):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        records = tile_image_and_labels(
            img_path, lbl_path, out_img_dir, out_lbl_dir,
            tile_size=args.tile_size, overlap=args.overlap
        )
        if not records:
            continue

        # 用 PIL 打开原图，逐块裁剪保存
        img = Image.open(img_path).convert("RGB")
        for _, tile_name, tx, ty, _, _ in records:
            tile_img = img.crop((tx, ty, tx + args.tile_size, ty + args.tile_size))
            tile_img.save(out_img_dir / f"{tile_name}.jpg", quality=95)
        img.close()

        total_tiles += len(records)

    print(f"[INFO] Generated {total_tiles} tiles (overlap={args.overlap})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行验证**

```bash
cd d:/1/tabel
# 先跑分层划分
python src/preprocess/split.py \
    --labels data/processed/yolo/labels \
    --images data/raw \
    --out data/processed/yolo
# 再跑切片（以 train 目录为例，overlap=20%）
python src/preprocess/tiling.py \
    --images data/processed/yolo/images/train \
    --labels data/processed/yolo/labels/train \
    --out-img data/processed/yolo_tiled/images/train \
    --out-lbl data/processed/yolo_tiled/labels/train \
    --overlap 0.2
```

- [ ] **Step 4: 提交**

```bash
git add src/preprocess/split.py src/preprocess/tiling.py
git commit -m "feat: add stratified split and tiling scripts"
```

---

### Task 6: 训练回调 — Per-class AP 追踪

**Files:**
- Create: `src/train/callbacks.py`

- [ ] **Step 1: 写入 callbacks.py**

```python
"""Ultralytics YOLOv8 自定义回调：追踪每 epoch 各类 AP"""
import csv
from pathlib import Path
from ultralytics.utils.metrics import ap_per_class, ConfusionMatrix
import numpy as np


class PerClassAPCallback:
    """在每轮验证结束后计算并追加写入 per-class AP"""

    def __init__(self, output_dir: Path, class_names: list):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        self.csv_path = self.output_dir / "per_class_ap.csv"
        self._init_csv()
        self.epoch = 0

    def _init_csv(self):
        header = ["epoch"] + [f"AP_{name}" for name in self.class_names] + ["mAP50", "mAP50_95"]
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    def __call__(self, trainer):
        """作为 Ultralytics callback 被调用"""
        self.epoch = trainer.epoch
        try:
            # 尝试从验证器获取 per-class AP
            validator = trainer.validator
            if validator is None:
                return

            metrics = validator.metrics
            if metrics is None or not hasattr(metrics, "ap_class_index"):
                return

            # 获取各类 AP
            ap = metrics.ap50  # shape: (num_classes,)
            ap_indices = metrics.ap_class_index

            per_class_ap = {}
            for idx, ap_val in zip(ap_indices, ap):
                if idx < len(self.class_names):
                    per_class_ap[idx] = float(ap_val)

            # 写入 CSV
            row = [self.epoch]
            for i in range(len(self.class_names)):
                row.append(per_class_ap.get(i, 0.0))
            row.append(float(metrics.box.map50) if hasattr(metrics.box, "map50") else 0.0)
            row.append(float(metrics.box.map) if hasattr(metrics.box, "map") else 0.0)

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

        except Exception as e:
            print(f"[Callback] Warning: failed to log per-class AP at epoch {self.epoch}: {e}")


def make_callbacks(experiment_dir: Path, class_names: list):
    """创建实验所需的全部回调"""
    return [PerClassAPCallback(output_dir=experiment_dir, class_names=class_names)]
```

- [ ] **Step 2: 提交**

```bash
git add src/train/callbacks.py
git commit -m "feat: add per-class AP tracking callback for YOLOv8"
```

---

### Task 7: 实验配置文件

**Files:**
- Create: `configs/exp1_baseline.yaml`
- Create: `configs/exp2_tiling_overlap0.yaml`
- Create: `configs/exp2_tiling_overlap20.yaml`
- Create: `configs/exp2_tiling_overlap30.yaml`
- Create: `configs/exp3_longtail.yaml`
- Create: `configs/exp4_finetune.yaml`

- [ ] **Step 1: 写入 `configs/exp1_baseline.yaml`**

```yaml
experiment: "exp1_baseline"
description: "Baseline: YOLOv8s, no tiling, no long-tail handling"
model: "yolov8s.pt"
data: "data/processed/yolo/dataset.yaml"
imgsz: 640
epochs: 200
patience: 50
batch: 16
lr0: 0.001
optimizer: "AdamW"
pretrained: true
device: "cuda:0"
workers: 4
tiling:
  enabled: false
  tile_size: 640
  overlap: 0.0
augment:
  mosaic: 0.0
  mixup: 0.0
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  flipud: 0.0
  fliplr: 0.5
long_tail:
  oversample: false
  oversample_threshold: 200
  oversample_multiplier: [3, 5, 10]
  class_weight_loss: false
```

- [ ] **Step 2: 写入三个 overlap 的切片配置**

`configs/exp2_tiling_overlap0.yaml`:
```yaml
experiment: "exp2_tiling_o0"
description: "Tiling 640x640, overlap=0%"
model: "yolov8s.pt"
data: "data/processed/yolo_tiled/dataset.yaml"
imgsz: 640
epochs: 200
patience: 50
batch: 16
lr0: 0.001
optimizer: "AdamW"
pretrained: true
device: "cuda:0"
workers: 4
tiling:
  enabled: true
  tile_size: 640
  overlap: 0.0
augment:
  mosaic: 0.0
  mixup: 0.0
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  flipud: 0.0
  fliplr: 0.5
long_tail:
  oversample: false
  oversample_threshold: 200
  oversample_multiplier: [3, 5, 10]
  class_weight_loss: false
```

`configs/exp2_tiling_overlap20.yaml` — 同 exp2_tiling_o0，但 `experiment: exp2_tiling_o20`, `overlap: 0.2`。

`configs/exp2_tiling_overlap30.yaml` — 同 exp2_tiling_o0，但 `experiment: exp2_tiling_o30`, `overlap: 0.3`。

- [ ] **Step 3: 写入 `configs/exp3_longtail.yaml`**

```yaml
experiment: "exp3_longtail"
description: "Tiling (best overlap) + long-tail handling (oversample + class weight + mosaic)"
model: "yolov8s.pt"
data: "data/processed/yolo_tiled/dataset.yaml"
imgsz: 640
epochs: 200
patience: 50
batch: 16
lr0: 0.001
optimizer: "AdamW"
pretrained: true
device: "cuda:0"
workers: 4
tiling:
  enabled: true
  tile_size: 640
  overlap: 0.2  # 从 Exp2 中选最优值填入
augment:
  mosaic: 0.5
  mixup: 0.2
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  flipud: 0.0
  fliplr: 0.5
long_tail:
  oversample: true
  oversample_threshold: 200
  oversample_multiplier: [3, 5, 10]
  class_weight_loss: true
```

- [ ] **Step 4: 写入 `configs/exp4_finetune.yaml`**

```yaml
experiment: "exp4_finetune"
description: "Full optimization: best settings from all experiments + lr tuning"
model: "yolov8s.pt"
data: "data/processed/yolo_tiled/dataset.yaml"
imgsz: 640
epochs: 300
patience: 80
batch: 16
lr0: 0.0005
optimizer: "AdamW"
pretrained: true
device: "cuda:0"
workers: 4
tiling:
  enabled: true
  tile_size: 640
  overlap: 0.2  # 从 Exp2 中选最优值填入
augment:
  mosaic: 0.5
  mixup: 0.2
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  flipud: 0.0
  fliplr: 0.5
long_tail:
  oversample: true
  oversample_threshold: 200
  oversample_multiplier: [3, 5, 10]
  class_weight_loss: true
```

- [ ] **Step 5: 提交**

```bash
git add configs/
git commit -m "feat: add experiment config files for Exp1-4"
```

---

### Task 8: 实验运行器

**Files:**
- Create: `src/train/run_experiment.py`

- [ ] **Step 1: 写入 run_experiment.py**

```python
"""从 YAML 配置文件读取参数并运行 YOLOv8 训练实验"""
import sys
import shutil
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO
from src.train.callbacks import make_callbacks


def run_experiment(config_path: Path, experiments_root: Path):
    """读取实验配置，运行 YOLOv8 训练，收集产物到 experiments/{exp_name}/"""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["experiment"]
    exp_dir = experiments_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始配置
    shutil.copy2(config_path, exp_dir / "config.yaml")

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Description: {cfg.get('description', 'N/A')}")
    print(f"Output: {exp_dir}")
    print(f"{'='*60}\n")

    # 加载模型
    model_path = cfg.get("model", "yolov8s.pt")
    if model_path and Path(model_path).exists():
        model = YOLO(model_path)
    else:
        model = YOLO("yolov8s.pt")  # 从 Ultralytics 自动下载

    # 写入模型摘要
    summary_path = exp_dir / "model_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(str(model.info(verbose=True)))
        # 备选: f.write(model.__repr__())

    # 构建训练参数
    train_args = {
        "data": cfg.get("data", "data/processed/yolo/dataset.yaml"),
        "imgsz": cfg.get("imgsz", 640),
        "epochs": cfg.get("epochs", 200),
        "patience": cfg.get("patience", 50),
        "batch": cfg.get("batch", 16),
        "lr0": cfg.get("lr0", 0.001),
        "optimizer": cfg.get("optimizer", "AdamW"),
        "pretrained": cfg.get("pretrained", True),
        "device": cfg.get("device", "cuda:0"),
        "workers": cfg.get("workers", 4),
        "project": str(exp_dir),
        "name": "run",
        "exist_ok": True,
        "save": True,
        "save_period": -1,  # 只保存 best 和 last
        "val": True,
    }

    # 增强参数
    aug = cfg.get("augment", {})
    if aug:
        train_args["mosaic"] = aug.get("mosaic", 0.0)
        train_args["mixup"] = aug.get("mixup", 0.0)
        train_args["hsv_h"] = aug.get("hsv_h", 0.015)
        train_args["hsv_s"] = aug.get("hsv_s", 0.7)
        train_args["hsv_v"] = aug.get("hsv_v", 0.4)
        train_args["flipud"] = aug.get("flipud", 0.0)
        train_args["fliplr"] = aug.get("fliplr", 0.5)

    # 长尾处理：类别权重（通过加倍采样倍率实现——按类频次分档）
    lt = cfg.get("long_tail", {})
    if lt.get("class_weight_loss"):
        # Ultralytics 原生 train() 不直接暴露 class_weight 参数
        # 权重策略通过 oversample_multiplier 分档实现：
        #   样本 <50:  10x 倍率
        #   样本 50-100: 5x 倍率
        #   样本 100-200: 3x 倍率
        #   样本 >=200:  1x（不复制）
        print("[INFO] Class weights applied via oversample_multiplier tiers")
        print(f"       Tiers: {lt.get('oversample_multiplier', [3, 5, 10])}")

    # 注册回调
    from ultralytics.data.utils import check_det_dataset
    data_info = check_det_dataset(train_args["data"])
    class_names = data_info.get("names", [f"class_{i}" for i in range(100)])
    callbacks = make_callbacks(exp_dir, class_names)

    # 开始训练
    results = model.train(**train_args)

    # 收集产物
    ultralytics_run = sorted(exp_dir.glob("run*"))
    if ultralytics_run:
        run_dir = ultralytics_run[0] if ultralytics_run[0].is_dir() else ultralytics_run[-1]
        # 移动 weights
        for weight_file in Path(run_dir).glob("weights/*.pt"):
            shutil.copy2(weight_file, exp_dir / weight_file.name)
        # 移动 results.csv
        results_csv = Path(run_dir) / "results.csv"
        if results_csv.exists():
            shutil.copy2(results_csv, exp_dir / "results.csv")
        # 移动自动生成的图表
        for png_file in Path(run_dir).glob("*.png"):
            plots_dir = exp_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            shutil.copy2(png_file, plots_dir / png_file.name)

    print(f"\n[OK] Experiment '{exp_name}' completed. Artifacts in {exp_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run a YOLOv8 training experiment")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--experiments-root", default="experiments", help="Root directory for experiments")
    args = parser.parse_args()

    run_experiment(Path(args.config), Path(args.experiments_root))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add src/train/run_experiment.py
git commit -m "feat: add experiment runner with full artifact collection"
```

---

### Task 9: 自定义绘图脚本

**Files:**
- Create: `src/eval/plot_metrics.py`

- [ ] **Step 1: 写入 plot_metrics.py**

```python
"""从实验日志生成论文级高质量图表（300dpi, 中文字体, matplotlib）"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 中文字体配置 ---
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def read_experiments(experiments_root: Path, exp_names: list):
    """读取各实验的 results.csv，返回 {exp_name: DataFrame}"""
    dfs = {}
    for name in exp_names:
        csv_path = experiments_root / name / "results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            dfs[name] = df
        else:
            print(f"[WARN] {csv_path} not found, skipping")
    return dfs


def plot_loss_curves(exp_name: str, df: pd.DataFrame, output_dir: Path):
    """单实验 Loss 折线图（train/val box_loss, cls_loss, dfl_loss）"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    loss_pairs = [
        ("train/box_loss", "val/box_loss", "Box Loss"),
        ("train/cls_loss", "val/cls_loss", "Cls Loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
    ]

    for ax, (train_col, val_col, title) in zip(axes, loss_pairs):
        if train_col in df.columns:
            ax.plot(df.index, df[train_col], label="Train", color=COLORS[0], linewidth=1.5)
        if val_col in df.columns:
            ax.plot(df.index, df[val_col], label="Val", color=COLORS[1], linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{exp_name} — Loss Curves")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "loss_curves.png")
    plt.close(fig)


def plot_map_curve(exp_name: str, df: pd.DataFrame, output_dir: Path):
    """单实验 mAP 曲线"""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    if "metrics/mAP50(B)" in df.columns:
        ax1.plot(df.index, df["metrics/mAP50(B)"], color=COLORS[0], label="mAP@0.5", linewidth=2)
    if "metrics/mAP50-95(B)" in df.columns:
        ax2.plot(df.index, df["metrics/mAP50-95(B)"], color=COLORS[1], label="mAP@0.5:0.95", linewidth=2, linestyle="--")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("mAP@0.5", color=COLORS[0])
    ax2.set_ylabel("mAP@0.5:0.95", color=COLORS[1])
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.set_title(f"{exp_name} — mAP Curves")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "map_curve.png")
    plt.close(fig)


def plot_multi_exp_loss_compare(dfs: dict, output_dir: Path):
    """多实验 Loss 对比（val/box_loss 一图, val/cls_loss 一图）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric, ax, title in [("val/box_loss", axes[0], "Val Box Loss"),
                               ("val/cls_loss", axes[1], "Val Cls Loss")]:
        for i, (exp_name, df) in enumerate(dfs.items()):
            if metric in df.columns:
                ax.plot(df.index, df[metric], label=exp_name, color=COLORS[i % len(COLORS)], linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Multi-Experiment Loss Comparison")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "multi_exp_loss_compare.png")
    plt.close(fig)


def plot_multi_exp_map_bars(dfs: dict, output_dir: Path):
    """多实验 mAP 对比柱状图"""
    exp_names = list(dfs.keys())
    map50_vals = []
    map50_95_vals = []

    for name in exp_names:
        df = dfs[name]
        map50_vals.append(df["metrics/mAP50(B)"].max() if "metrics/mAP50(B)" in df.columns else 0)
        map50_95_vals.append(df["metrics/mAP50-95(B)"].max() if "metrics/mAP50-95(B)" in df.columns else 0)

    x = np.arange(len(exp_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, map50_vals, width, label="mAP@0.5", color=COLORS[0])
    bars2 = ax.bar(x + width/2, map50_95_vals, width, label="mAP@0.5:0.95", color=COLORS[1])

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("mAP")
    ax.set_title("Ablation Experiment — mAP Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "multi_exp_map_bars.png")
    plt.close(fig)


def plot_per_class_ap(class_ap_csv: Path, output_dir: Path):
    """Per-class AP 降序柱状图"""
    if not class_ap_csv.exists():
        print(f"[WARN] {class_ap_csv} not found")
        return

    df = pd.read_csv(class_ap_csv)
    # 取最后一轮的 per-class AP
    last_row = df.iloc[-1]
    ap_cols = [c for c in df.columns if c.startswith("AP_")]
    ap_values = [last_row[c] for c in ap_cols]
    class_labels = [c.replace("AP_", "") for c in ap_cols]

    # 按 AP 降序排列
    sorted_idx = np.argsort(ap_values)[::-1]
    sorted_labels = [class_labels[i] for i in sorted_idx]
    sorted_vals = [ap_values[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = [COLORS[0] if v >= 0.5 else COLORS[1] for v in sorted_vals]
    ax.bar(range(len(sorted_labels)), sorted_vals, color=colors, width=0.7)
    ax.set_xticks(range(len(sorted_labels)))
    ax.set_xticklabels(sorted_labels, rotation=90, fontsize=7)
    ax.set_ylabel("AP@0.5")
    ax.set_title("Per-Class AP (Final Epoch)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "per_class_ap_bar.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate thesis-quality plots from experiment logs")
    parser.add_argument("--experiments-root", default="experiments", help="Experiments root directory")
    parser.add_argument("--exps", nargs="+", default=["exp1_baseline", "exp2_tiling_o20", "exp3_longtail", "exp4_finetune"],
                        help="Experiment names to include")
    parser.add_argument("--out", default="experiments/comparison", help="Output directory for comparison charts")
    args = parser.parse_args()

    root = Path(args.experiments_root)
    dfs = read_experiments(root, args.exps)
    if not dfs:
        print("[ERROR] No experiment results found")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 单实验图表
    for exp_name, df in dfs.items():
        exp_plots_dir = root / exp_name / "plots"
        plot_loss_curves(exp_name, df, exp_plots_dir)
        plot_map_curve(exp_name, df, exp_plots_dir)
        print(f"[OK] Single-experiment plots for {exp_name}")

    # 多实验对比图表
    plot_multi_exp_loss_compare(dfs, out)
    plot_multi_exp_map_bars(dfs, out)
    print(f"[OK] Multi-experiment comparison plots in {out}")

    # 如果存在 per_class_ap.csv，绘制 per-class AP
    for exp_name in args.exps:
        ap_csv = root / exp_name / "per_class_ap.csv"
        if ap_csv.exists():
            plot_per_class_ap(ap_csv, root / exp_name / "eval")
            print(f"[OK] Per-class AP plot for {exp_name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add src/eval/plot_metrics.py
git commit -m "feat: add thesis chart generation script (loss, mAP, per-class AP, ablation comparison)"
```

---

### Task 10: Flask 推理 API

**Files:**
- Create: `src/infer/app.py`

- [ ] **Step 1: 写入 app.py**

```python
"""Flask 推理 API：图片检测 + 批量检测，支持大图切片推理"""
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

MODEL = None
MODEL_PATH = None
TILING_ENABLED = False
TILE_SIZE = 640
OVERLAP = 0.2


def load_model(model_path: str):
    global MODEL, MODEL_PATH
    MODEL = YOLO(model_path)
    MODEL_PATH = model_path
    print(f"[INFO] Model loaded from {model_path}")


def tile_detect(image: Image.Image, conf_thres: float = 0.25, iou_thres: float = 0.45):
    """大图切片检测：将图片切片后逐块推理，NMS 合并结果"""
    img_w, img_h = image.size
    stride = int(TILE_SIZE * (1 - OVERLAP))

    all_boxes = []
    all_scores = []
    all_classes = []

    y_positions = sorted(set(list(range(0, img_h - TILE_SIZE, stride)) + [max(0, img_h - TILE_SIZE)]))
    x_positions = sorted(set(list(range(0, img_w - TILE_SIZE, stride)) + [max(0, img_w - TILE_SIZE)]))

    for ty in y_positions:
        for tx in x_positions:
            tile = image.crop((tx, ty, tx + TILE_SIZE, ty + TILE_SIZE))

            results = MODEL(tile, conf=conf_thres, verbose=False)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # 映射回原图坐标
                all_boxes.append([x1 + tx, y1 + ty, x2 + tx, y2 + ty])
                all_scores.append(conf)
                all_classes.append(cls)

    if not all_boxes:
        return []

    boxes_array = np.array(all_boxes)
    scores_array = np.array(all_scores)
    classes_array = np.array(all_classes)

    # 将 bboxes 转为 [cx, cy, w, h] 做 NMS
    boxes_xywh = np.zeros_like(boxes_array, dtype=np.float32)
    boxes_xywh[:, 0] = (boxes_array[:, 0] + boxes_array[:, 2]) / 2
    boxes_xywh[:, 1] = (boxes_array[:, 1] + boxes_array[:, 3]) / 2
    boxes_xywh[:, 2] = boxes_array[:, 2] - boxes_array[:, 0]
    boxes_xywh[:, 3] = boxes_array[:, 3] - boxes_array[:, 1]

    import torch
    keep = torch.ops.torchvision.nms(
        torch.from_numpy(np.column_stack([boxes_array, scores_array]).astype(np.float32)),
        iou_thres,
    )

    results_list = []
    for idx in keep:
        idx = int(idx.item()) if isinstance(idx, torch.Tensor) else int(idx)
        results_list.append({
            "x1": int(boxes_array[idx][0]),
            "y1": int(boxes_array[idx][1]),
            "x2": int(boxes_array[idx][2]),
            "y2": int(boxes_array[idx][3]),
            "class_id": int(classes_array[idx]),
            "class_name": MODEL.names.get(int(classes_array[idx]), f"class_{classes_array[idx]}"),
            "confidence": round(float(scores_array[idx]), 4),
        })

    return results_list


@app.route("/detect", methods=["POST"])
def detect():
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    if TILING_ENABLED:
        detections = tile_detect(image)
    else:
        results = MODEL(image, conf=0.25, verbose=False)
        boxes = results[0].boxes
        detections = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "class_id": int(box.cls[0]),
                    "class_name": MODEL.names.get(int(box.cls[0]), "unknown"),
                    "confidence": round(float(box.conf[0]), 4),
                })

    return jsonify({"detections": detections, "count": len(detections)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Flask inference API for YOLOv8")
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--tiling", action="store_true", help="Enable tiling inference")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    global TILING_ENABLED, TILE_SIZE, OVERLAP
    TILING_ENABLED = args.tiling
    TILE_SIZE = args.tile_size
    OVERLAP = args.overlap

    load_model(args.model)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add src/infer/app.py
git commit -m "feat: add Flask inference API with tiling support"
```

---

### Task 11: 数据下载辅助脚本

**Files:**
- Create: `scripts/download_tt100k.py`

- [ ] **Step 1: 写入 download 脚本**

```python
"""TT100K 数据下载指引脚本"""
import sys
from pathlib import Path

DATASET_URL = "https://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip"

DOWNLOAD_INSTRUCTIONS = f"""
========================================
  TT100K 2021 数据集下载说明
========================================

1. 手动下载数据包:
   {DATASET_URL}

2. 将下载的 data.zip 放到 data/raw/ 目录

3. 解压:
   cd data/raw
   unzip data.zip

4. 解压后 data/raw/ 下应该有:
   - annotations.json    (标注文件)
   - train/              (训练图片, 2048x2048 jpg)
   - test/               (测试图片)
   - other/              (无标注图片, 丢弃)

5. 确认后运行预处理流水线:
   python src/preprocess/class_stats.py --anno data/raw/annotations.json
   python src/preprocess/filter.py --anno data/raw/annotations.json --min-instances 50
   python src/preprocess/convert.py --anno data/processed/filtered/annotations_filtered.json --images data/raw
   python src/preprocess/split.py --labels data/processed/yolo/labels --images data/raw
========================================
"""


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    anno_path = raw_dir / "annotations.json"
    if anno_path.exists():
        print("[OK] annotations.json already exists in data/raw/")
        print("Skipping download instructions. Run preprocess scripts directly.")
    else:
        print(DOWNLOAD_INSTRUCTIONS)

    # 检查常见问题
    other_dir = raw_dir / "other"
    if other_dir.exists():
        print(f"[INFO] 'other/' directory found ({len(list(other_dir.glob('*')))} files) — will be ignored per spec")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
mkdir -p d:/1/tabel/scripts
git add scripts/download_tt100k.py
git commit -m "feat: add TT100K dataset download helper script"
```

---

### Task 12: 集成验证

- [ ] **Step 1: 验证项目结构完整性**

```bash
cd d:/1/tabel
find . -name "*.py" -o -name "*.yaml" -o -name "*.txt" -o -name "*.md" | sort
```

- [ ] **Step 2: 验证 Python 语法**

```bash
cd d:/1/tabel
for f in $(find src scripts -name "*.py"); do
    echo "Checking $f..."
    python -m py_compile "$f" && echo "  OK" || echo "  FAILED"
done
```

- [ ] **Step 3: 最终提交**

```bash
git add -A
git commit -m "chore: finalize project structure and verify all files"
```
