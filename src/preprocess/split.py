"""分层划分训练/验证/测试集，保持各类别比例一致"""
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split


def split_dataset(labels_dir: Path, images_dir: Path, output_dir: Path,
                  train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01

    txt_files = sorted(labels_dir.glob("*.txt"))
    print(f"[INFO] {len(txt_files)} label files found")

    # 统计每张图中各类别实例数 → 取主导类作为分层标签
    file_class_counts = {}
    for txt in txt_files:
        counts = defaultdict(int)
        content = txt.read_text(encoding="utf-8").strip()
        if content:
            for line in content.split("\n"):
                if line.strip():
                    cls_id = int(line.split()[0])
                    counts[cls_id] += 1
        file_class_counts[txt.stem] = dict(counts)

    stems = sorted(file_class_counts.keys())
    stratify_labels = []
    for s in stems:
        counts = file_class_counts[s]
        dominant_class = max(counts, key=counts.get) if counts else 0
        stratify_labels.append(dominant_class)

    # 先分 test
    train_val_stems, test_stems = train_test_split(
        stems, test_size=test_ratio, random_state=42, stratify=stratify_labels
    )
    # 再分 train / val
    tv_indices = [stems.index(s) for s in train_val_stems]
    tv_stratify = [stratify_labels[i] for i in tv_indices]
    train_stems, val_stems = train_test_split(
        train_val_stems,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=42,
        stratify=tv_stratify,
    )

    # 查找图片后缀
    def find_image(stem, img_dir):
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            for f in img_dir.rglob(stem + ext):
                return f
        return None

    splits = [("train", train_stems), ("val", val_stems), ("test", test_stems)]

    for split_name, split_stems in splits:
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for stem in split_stems:
            src_img = find_image(stem, images_dir)
            if src_img:
                shutil.copy2(src_img, img_out / f"{stem}{src_img.suffix}")
            else:
                print(f"[WARN] Image not found: {stem}")

            src_lbl = labels_dir / f"{stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, lbl_out / f"{stem}.txt")

        print(f"[INFO] {split_name}: {len(split_stems)} images")

    # 更新 dataset.yaml 为绝对路径
    yaml_path = output_dir / "dataset.yaml"
    if yaml_path.exists():
        content = yaml_path.read_text(encoding="utf-8")
        content = content.replace(
            "images/train",
            (output_dir / "images" / "train").absolute().as_posix()
        )
        content = content.replace(
            "images/val",
            (output_dir / "images" / "val").absolute().as_posix()
        )
        content = content.replace(
            "images/test",
            (output_dir / "images" / "test").absolute().as_posix()
        )
        yaml_path.write_text(content, encoding="utf-8")
        print(f"[INFO] Updated {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Stratified train/val/test split")
    parser.add_argument("--labels", required=True, help="Directory of YOLO txt labels")
    parser.add_argument("--images", required=True, help="Root directory of original images")
    parser.add_argument("--out", default="data/processed/yolo", help="Output root directory")
    args = parser.parse_args()

    split_dataset(Path(args.labels), Path(args.images), Path(args.out))


if __name__ == "__main__":
    main()
