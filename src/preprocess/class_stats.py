"""TT100K 类别分布统计与 bbox 尺寸分析"""
import json
import csv
import argparse
from pathlib import Path
from collections import Counter


def parse_annotations(anno_path: Path):
    """读取 TT100K JSON，返回 [(image_name, [objects]), ...]"""
    with open(anno_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for key, val in data.items():
        img_name = Path(key).name
        objects = val.get("objects", val) if isinstance(val, dict) else val
        if isinstance(objects, dict):
            objects = objects.get("objects", [])
        results.append((img_name, objects))

    print(f"[INFO] {len(results)} images loaded, total objects: {sum(len(objs) for _, objs in results)}")
    return results


def compute_stats(annotations: list, output_dir: Path):
    """计算类别实例数分布和 bbox 尺寸分布"""
    class_counter = Counter()
    size_bins = {"small": 0, "medium": 0, "large": 0}

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

    # 汇总
    total_classes = len(class_counter)
    head = sum(1 for _, c in rows if c >= 500)
    mid = sum(1 for _, c in rows if 100 <= c < 500)
    low1 = sum(1 for _, c in rows if 50 <= c < 100)
    low2 = sum(1 for _, c in rows if 10 <= c < 50)
    tail = sum(1 for _, c in rows if 1 <= c < 10)
    zero = sum(1 for _, c in rows if c == 0)
    total_instances = sum(size_bins.values())

    summary = f"""
=== TT100K Class Distribution Summary ===
Total classes: {total_classes}
  >= 500 instances: {head}
  100-499 instances: {mid}
  50-99 instances:   {low1}
  10-49 instances:   {low2}
  1-9 instances:     {tail}
  0 instances:       {zero}
Classes with >=50 instances: {head + mid + low1}

Bbox size distribution (in original 2048px):
  Small  (<32x32):  {size_bins['small']:>6}  ({size_bins['small']/total_instances*100:.1f}%)
  Medium (32-96):   {size_bins['medium']:>6}  ({size_bins['medium']/total_instances*100:.1f}%)
  Large  (>96x96):  {size_bins['large']:>6}  ({size_bins['large']/total_instances*100:.1f}%)
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
