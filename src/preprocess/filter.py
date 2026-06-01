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

    class_names = data.get("types", [])
    imgs = data.get("imgs", {})

    # 统计每个类别的实例数
    class_counter = Counter()
    for img_id, img_info in imgs.items():
        for obj in img_info.get("objects", []):
            cat = obj.get("category", "")
            class_counter[cat] += 1

    print(f"[INFO] {len(class_names)} types, {len(class_counter)} classes with instances")

    # 筛选出实例数 >= min_instances 的类
    valid_classes = sorted([c for c, n in class_counter.items() if n >= min_instances],
                           key=lambda c: -class_counter[c])
    class_map = {cls: idx for idx, cls in enumerate(valid_classes)}

    print(f"[INFO] {len(valid_classes)} classes with >= {min_instances} instances")

    # 保存类别映射表
    map_path = output_dir / "class_mapping.csv"
    with open(map_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_class", "new_id", "instance_count"])
        for cls_name, new_id in class_map.items():
            writer.writerow([cls_name, new_id, class_counter[cls_name]])
    print(f"[INFO] Class mapping saved to {map_path}")

    # 过滤标注：只保留属于有效类别的对象
    filtered_imgs = {}
    total_kept = 0
    total_filtered = 0
    for img_id, img_info in imgs.items():
        kept_objects = []
        for obj in img_info.get("objects", []):
            cat = obj.get("category", "")
            if cat in class_map:
                obj_copy = dict(obj)
                obj_copy["new_id"] = class_map[cat]
                kept_objects.append(obj_copy)
                total_kept += 1
            else:
                total_filtered += 1

        if kept_objects:
            filtered_imgs[img_id] = {
                "path": img_info["path"],
                "id": img_info.get("id", img_id),
                "objects": kept_objects,
            }

    print(f"[INFO] Kept {total_kept} objects, filtered out {total_filtered}")

    # 保存过滤后的标注（保持原格式）
    filtered_data = {
        "types": class_names,
        "imgs": filtered_imgs,
    }
    filtered_path = output_dir / "annotations_filtered.json"
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Filtered annotations saved to {filtered_path} ({len(filtered_imgs)} images)")

    return class_map


def main():
    parser = argparse.ArgumentParser(description="Filter TT100K classes by instance count")
    parser.add_argument("--anno", required=True, help="Path to annotations_all.json")
    parser.add_argument("--min-instances", type=int, default=100, help="Minimum instances per class (default: 100)")
    parser.add_argument("--out", default="data/processed/filtered", help="Output directory")
    args = parser.parse_args()

    filter_classes(Path(args.anno), args.min_instances, Path(args.out))


if __name__ == "__main__":
    main()
