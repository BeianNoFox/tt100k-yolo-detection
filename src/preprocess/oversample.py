"""长尾加倍采样：对低频类图片按档位复制，产出 oversampled 数据集"""
import argparse
import shutil
from pathlib import Path
from collections import Counter
import yaml


def oversample_dataset(images_dir: Path, labels_dir: Path, output_dir: Path,
                       threshold: int = 200, multipliers: list = None):
    if multipliers is None:
        multipliers = [3, 5, 10]

    output_img = output_dir / "images" / "train"
    output_lbl = output_dir / "labels" / "train"
    output_img.mkdir(parents=True, exist_ok=True)
    output_lbl.mkdir(parents=True, exist_ok=True)

    # 1. 统计各类别实例数
    class_counter = Counter()
    img_classes = {}  # {stem: [class_ids in this image]}
    for lbl in sorted(labels_dir.glob("*.txt")):
        stem = lbl.stem
        classes = []
        for line in lbl.read_text().strip().split("\n"):
            if line.strip():
                cls_id = int(line.split()[0])
                class_counter[cls_id] += 1
                classes.append(cls_id)
        img_classes[stem] = list(set(classes))

    num_classes = len(class_counter)
    print(f"[INFO] {num_classes} classes, {len(img_classes)} images")

    # 2. 确定每类的复制倍率：<50→10x, 50-100→5x, 100-200→3x, >=200→1x
    class_multiplier = {}
    for cls_id, count in class_counter.items():
        if count < 50:
            class_multiplier[cls_id] = multipliers[2]  # 10x
        elif count < 100:
            class_multiplier[cls_id] = multipliers[1]  # 5x
        elif count < threshold:
            class_multiplier[cls_id] = multipliers[0]  # 3x
        else:
            class_multiplier[cls_id] = 1  # no oversample

    # 3. 对每张图，取图中最稀有的类的倍率作为该图的倍率
    total_src = 0
    total_dst = 0
    for stem, cls_ids in img_classes.items():
        max_mult = max(class_multiplier[c] for c in cls_ids)

        src_lbl = labels_dir / f"{stem}.txt"
        src_img = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                src_img = candidate
                break

        if not src_img:
            print(f"[WARN] Image not found: {stem}")
            continue

        total_src += 1

        # 复制原图 + 标注
        for dup_idx in range(max_mult):
            suffix = f"_x{dup_idx}" if dup_idx > 0 else ""
            dst_stem = f"{stem}{suffix}"
            shutil.copy2(src_img, output_img / f"{dst_stem}{src_img.suffix}")
            shutil.copy2(src_lbl, output_lbl / f"{dst_stem}.txt")
            total_dst += 1

    print(f"[INFO] {total_src} source images → {total_dst} oversampled images "
          f"({(total_dst/total_src - 1)*100:.1f}% increase)")

    # 统计各类前后实例数对比
    new_counter = Counter()
    for lbl in sorted(output_lbl.glob("*.txt")):
        for line in lbl.read_text().strip().split("\n"):
            if line.strip():
                new_counter[int(line.split()[0])] += 1

    print("\n--- Top-10 gain (instance count change) ---")
    gains = []
    for cls_id in range(num_classes):
        old = class_counter.get(cls_id, 0)
        new = new_counter.get(cls_id, 0)
        gains.append((cls_id, old, new, new - old))
    gains.sort(key=lambda x: -x[3])
    for cls_id, old, new, delta in gains[:10]:
        print(f"  class {cls_id:3}: {old:>5} → {new:>5} (+{delta})")

    # 4. 复制 val 和 test（不增强）
    yolo_root = labels_dir.parent.parent  # data/processed/yolo (labels_dir = .../labels/train)
    for split in ["val", "test"]:
        src_img_dir = yolo_root / "images" / split
        src_lbl_dir = yolo_root / "labels" / split
        if not src_img_dir.exists():
            continue
        dst_img = output_dir / "images" / split
        dst_lbl = output_dir / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for f in src_img_dir.iterdir():
            shutil.copy2(f, dst_img / f.name)
        for f in src_lbl_dir.iterdir():
            shutil.copy2(f, dst_lbl / f.name)
        print(f"[INFO] Copied {split}: {sum(1 for _ in src_img_dir.iterdir())} images")

    # 5. 生成 dataset.yaml
    src_yaml = yolo_root / "dataset.yaml"
    if src_yaml.exists():
        with open(src_yaml) as f:
            ds_cfg = yaml.safe_load(f)
    else:
        ds_cfg = {"nc": num_classes, "names": [f"class_{i}" for i in range(num_classes)]}

    ds_cfg["path"] = str(output_dir.absolute().as_posix())
    ds_cfg["train"] = str(output_img.absolute().as_posix())
    ds_cfg["val"] = str((output_dir / "images" / "val").absolute().as_posix())
    if (output_dir / "images" / "test").exists():
        ds_cfg["test"] = str((output_dir / "images" / "test").absolute().as_posix())

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(ds_cfg, f)
    print(f"[INFO] dataset.yaml → {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Oversample low-frequency classes")
    parser.add_argument("--images", required=True, help="Train images directory")
    parser.add_argument("--labels", required=True, help="Train labels directory")
    parser.add_argument("--out", default="data/processed/yolo_oversampled", help="Output root")
    parser.add_argument("--threshold", type=int, default=200)
    args = parser.parse_args()

    oversample_dataset(Path(args.images), Path(args.labels), Path(args.out), args.threshold)


if __name__ == "__main__":
    main()
