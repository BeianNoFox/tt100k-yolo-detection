"""将 TT100K 过滤后 JSON 标注转换为 YOLO txt 格式"""
import json
import argparse
from pathlib import Path


def convert_to_yolo(filtered_json: Path, image_dir: Path, output_dir: Path):
    """
    输入过滤后的 JSON (TT100K 2021 格式):
      {"types": [...], "imgs": {"id": {"path": "...", "objects": [...]}}}
    输出：output_dir/ 下每张图片一个 .txt，格式：class_id cx cy w h（归一化）
    同时生成 dataset.yaml
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(filtered_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    imgs = data.get("imgs", {})

    # 从标注中收集所有 new_id → category 的映射
    class_names = {}
    for img_id, img_info in imgs.items():
        for obj in img_info["objects"]:
            new_id = obj["new_id"]
            cat = obj.get("category", "")
            if new_id not in class_names:
                class_names[new_id] = cat

    num_classes = len(class_names)
    print(f"[INFO] {num_classes} classes in filtered annotations")
    print(f"[INFO] Class names: {sorted(class_names.items())}")

    converted = 0
    skipped = 0
    for img_id, img_info in imgs.items():
        img_path = img_info["path"]  # e.g. "train/62627.jpg"
        full_img_path = image_dir / img_path

        if not full_img_path.exists():
            skipped += 1
            if skipped <= 3:
                print(f"[WARN] Image not found: {full_img_path}")
            continue

        # TT100K 图像均为 2048x2048
        img_w, img_h = 2048, 2048

        lines = []
        for obj in img_info["objects"]:
            new_id = obj["new_id"]
            bbox = obj["bbox"]  # {"xmin": ..., "ymin": ..., "xmax": ..., "ymax": ...}

            xmin = bbox["xmin"]
            ymin = bbox["ymin"]
            xmax = bbox["xmax"]
            ymax = bbox["ymax"]

            # 转为 YOLO 归一化格式
            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            lines.append(f"{new_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # 使用原始文件名（不含路径前缀）作为 txt 名
        txt_name = Path(img_path).stem + ".txt"
        (output_dir / txt_name).write_text("\n".join(lines), encoding="utf-8")
        converted += 1

    # 统计 train/test 来源
    train_count = sum(1 for v in imgs.values() if v["path"].startswith("train/"))
    test_count = sum(1 for v in imgs.values() if v["path"].startswith("test/"))
    other_count = sum(1 for v in imgs.values() if v["path"].startswith("other/"))

    print(f"[INFO] Converted {converted} images")
    print(f"[INFO] Skipped {skipped} (image file not found)")
    print(f"[INFO] Source split — train: {train_count}, test: {test_count}, other: {other_count}")

    # 生成 dataset.yaml（split 后的实际路径由 split.py 更新）
    names_sorted = [class_names[i] for i in range(num_classes)]
    yaml_content = f"""# TT100K filtered dataset (>=100 instances per class, 45 classes)
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
    parser.add_argument("--images", required=True, help="Root directory containing train/test/other subdirs")
    parser.add_argument("--out", default="data/processed/yolo/labels", help="Output directory for txt files")
    args = parser.parse_args()

    convert_to_yolo(Path(args.anno), Path(args.images), Path(args.out))


if __name__ == "__main__":
    main()
