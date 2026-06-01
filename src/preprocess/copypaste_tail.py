"""Copy-Paste 增强：抠出稀有类标志，随机贴到其他图上"""
import argparse
import shutil
import random
from pathlib import Path
from collections import Counter
import yaml
import cv2
import numpy as np


def copypaste_augment(images_dir: Path, labels_dir: Path, output_dir: Path,
                      threshold: int = 100, paste_per_class: int = 150):
    """
    对实例数 < threshold 的类别：抠出其所有标志实例，
    随机贴到其他图片上，每类至少生成 paste_per_class 个新样本。
    """
    output_img = output_dir / "images" / "train"
    output_lbl = output_dir / "labels" / "train"
    output_img.mkdir(parents=True, exist_ok=True)
    output_lbl.mkdir(parents=True, exist_ok=True)

    # 1. 加载所有图片和标注
    all_images = {}
    for img_path in sorted(images_dir.glob("*")):
        ext = img_path.suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = img_path.stem
        lbl_path = labels_dir / f"{stem}.txt"
        objects = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().split("\n"):
                if line.strip():
                    parts = line.split()
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    objects.append((cls_id, cx, cy, w, h))
        all_images[stem] = {"path": img_path, "objects": objects, "w": 2048, "h": 2048}

    # 2. 统计各类实例数，识别稀有类
    class_counter = Counter()
    for info in all_images.values():
        for cls_id, _, _, _, _ in info["objects"]:
            class_counter[cls_id] += 1

    rare_classes = {c for c, n in class_counter.items() if n < threshold}
    print(f"[INFO] {len(rare_classes)} rare classes (<{threshold} instances), generating {paste_per_class} pastes each")

    # 3. 收集稀有类的所有标志抠图
    class_crops = {c: [] for c in rare_classes}  # {class_id: [(crop_img, crop_w, crop_h), ...]}
    for info in all_images.values():
        img = cv2.imread(str(info["path"]))
        if img is None:
            continue
        for cls_id, cx, cy, w, h in info["objects"]:
            if cls_id in rare_classes:
                # 像素坐标
                px = int((cx - w / 2) * info["w"])
                py = int((cy - h / 2) * info["h"])
                pxx = int((cx + w / 2) * info["w"])
                pyy = int((cy + h / 2) * info["h"])
                crop = img[max(0, py):min(info["h"], pyy), max(0, px):min(info["w"], pxx)]
                if crop.size > 0:
                    class_crops[cls_id].append(crop)

    # 4. 先完整复制原始数据集
    print("[INFO] Copying original dataset...")
    for info in all_images.values():
        src = info["path"]
        shutil.copy2(src, output_img / src.name)
        lbl = labels_dir / f"{src.stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, output_lbl / lbl.name)

    # 5. 对每个稀有类，随机选择原图贴标志
    background_stems = list(all_images.keys())
    total_pasted = 0

    for cls_id in sorted(rare_classes):
        crops = class_crops.get(cls_id, [])
        if not crops:
            continue
        orig_count = class_counter.get(cls_id, 0)

        for paste_idx in range(paste_per_class):
            # 随机选背景图
            bg_stem = random.choice(background_stems)
            bg_info = all_images[bg_stem]
            bg_img = cv2.imread(str(bg_info["path"]))
            if bg_img is None:
                continue

            # 随机选一个该类的抠图
            crop = random.choice(crops)
            ch, cw = crop.shape[:2]

            # 随机缩放 0.7-1.3
            scale = random.uniform(0.7, 1.3)
            new_w = max(10, int(cw * scale))
            new_h = max(10, int(ch * scale))
            crop_resized = cv2.resize(crop, (new_w, new_h))

            # 随机位置（保证不越界）
            max_x = max(1, 2048 - new_w)
            max_y = max(1, 2048 - new_h)
            px = random.randint(0, max_x)
            py = random.randint(0, max_y)

            # Alpha 混合贴入（边缘自然）
            alpha = random.uniform(0.8, 0.95)  # 标志半透明
            roi = bg_img[py:py + new_h, px:px + new_w]
            blended = cv2.addWeighted(crop_resized, alpha, roi, 1 - alpha, 0)
            # 边缘羽化
            mask = np.zeros((new_h, new_w), dtype=np.float32)
            border = min(3, new_w // 4, new_h // 4)
            cv2.rectangle(mask, (border, border), (new_w - border, new_h - border), 1.0, -1)
            mask = cv2.GaussianBlur(mask, (border * 2 + 1, border * 2 + 1), 0)
            mask = mask[:, :, np.newaxis]
            bg_img[py:py + new_h, px:px + new_w] = (blended * mask + roi * (1 - mask)).astype(np.uint8)

            # 保存新图
            new_stem = f"{bg_stem}_cp{cls_id}_{paste_idx}"
            cv2.imwrite(str(output_img / f"{new_stem}.jpg"), bg_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # 写标注：原背景的所有标注 + 新贴的标志
            new_bbox = f"{cls_id} {(px + new_w/2)/2048:.6f} {(py + new_h/2)/2048:.6f} {new_w/2048:.6f} {new_h/2048:.6f}"
            orig_labels = []
            lbl_path = labels_dir / f"{bg_stem}.txt"
            if lbl_path.exists():
                for line in lbl_path.read_text().strip().split("\n"):
                    if line.strip():
                        orig_labels.append(line.strip())
            orig_labels.append(new_bbox)
            (output_lbl / f"{new_stem}.txt").write_text("\n".join(orig_labels))

            total_pasted += 1

    print(f"[INFO] Generated {total_pasted} copy-paste samples across {len(rare_classes)} rare classes")
    print(f"[INFO] Total images: {sum(1 for _ in output_img.glob('*'))}")

    # 6. 拷贝 val/test + dataset.yaml
    yolo_root = labels_dir.parent.parent
    yaml_src = yolo_root / "dataset.yaml"
    with open(yaml_src) as f:
        ds_cfg = yaml.safe_load(f)

    for split in ["val", "test"]:
        for sub in ["images", "labels"]:
            src_dir = yolo_root / sub / split
            dst_dir = output_dir / sub / split
            if not src_dir.exists():
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                shutil.copy2(f, dst_dir / f.name)
        print(f"[INFO] Copied {split}")

    ds_cfg["path"] = str(output_dir.absolute().as_posix())
    ds_cfg["train"] = str(output_img.absolute().as_posix())
    ds_cfg["val"] = str((output_dir / "images" / "val").absolute().as_posix())
    if (output_dir / "images" / "test").exists():
        ds_cfg["test"] = str((output_dir / "images" / "test").absolute().as_posix())
    with open(output_dir / "dataset.yaml", "w") as f:
        yaml.dump(ds_cfg, f)
    print(f"[INFO] Done → {output_dir / 'dataset.yaml'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="data/processed/yolo_copypaste")
    parser.add_argument("--threshold", type=int, default=100, help="Classes below this receive copy-paste")
    parser.add_argument("--paste-per-class", type=int, default=150, help="Paste instances per rare class")
    args = parser.parse_args()
    copypaste_augment(Path(args.images), Path(args.labels), Path(args.out),
                      args.threshold, args.paste_per_class)


if __name__ == "__main__":
    main()
