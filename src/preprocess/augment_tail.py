"""对极低频类进行强力数据增强：翻转、旋转、HSV、缩放、平移"""
import argparse
import shutil
import random
from pathlib import Path
from collections import Counter
import yaml
import cv2
import numpy as np


def augment_image(img: np.ndarray, seed: int) -> np.ndarray:
    """随机组合增强：水平翻转 + 旋转 + HSV + 亮度对比度 + 轻微模糊"""
    random.seed(seed)
    h, w = img.shape[:2]

    # 1. 水平翻转 (50%)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # 2. 旋转 ±20°
    angle = random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 3. HSV 偏移
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-15, 15)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.6, 1.4), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 4. 亮度对比度
    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-20, 20)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 5. 高斯噪声
    noise = np.random.normal(0, random.uniform(3, 10), img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # 6. 透视变换（轻微）
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    jitter = w * 0.05
    pts2 = np.float32([[random.uniform(0, jitter), random.uniform(0, jitter)],
                        [w - random.uniform(0, jitter), random.uniform(0, jitter)],
                        [random.uniform(0, jitter), h - random.uniform(0, jitter)],
                        [w - random.uniform(0, jitter), h - random.uniform(0, jitter)]])
    M_p = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M_p, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 7. 随机缩放 0.75-1.25
    scale = random.uniform(0.75, 1.25)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h))
    if scale > 1.0:
        # 裁剪回原尺寸
        x1 = (new_w - w) // 2
        y1 = (new_h - h) // 2
        img = img[y1:y1 + h, x1:x1 + w]
    else:
        # 填充黑边
        pad_w = (w - new_w) // 2
        pad_h = (h - new_h) // 2
        img = cv2.copyMakeBorder(img, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w,
                                 cv2.BORDER_REFLECT)

    return img


def oversample_with_augment(images_dir: Path, labels_dir: Path, output_dir: Path,
                             threshold: int = 200, multipliers: list = None):
    if multipliers is None:
        multipliers = [3, 5, 10]
    output_img = output_dir / "images" / "train"
    output_lbl = output_dir / "labels" / "train"
    output_img.mkdir(parents=True, exist_ok=True)
    output_lbl.mkdir(parents=True, exist_ok=True)

    # 1. 统计
    class_counter = Counter()
    img_classes = {}
    for lbl in sorted(labels_dir.glob("*.txt")):
        stem = lbl.stem
        classes = []
        for line in lbl.read_text().strip().split("\n"):
            if line.strip():
                cls_id = int(line.split()[0])
                class_counter[cls_id] += 1
                classes.append(cls_id)
        if classes:
            img_classes[stem] = list(set(classes))

    print(f"[INFO] {len(img_classes)} images, {len(class_counter)} classes")

    # 2. 倍率：<10→skip (用增强), 10-50→10x, 50-100→5x, 100-200→3x, >=200→1x
    class_multiplier = {}
    ultra_low = set()
    for cls_id, count in class_counter.items():
        if count < 10:
            class_multiplier[cls_id] = 1
            ultra_low.add(cls_id)
        elif count < 50:
            class_multiplier[cls_id] = multipliers[2]
        elif count < 100:
            class_multiplier[cls_id] = multipliers[1]
        elif count < threshold:
            class_multiplier[cls_id] = multipliers[0]
        else:
            class_multiplier[cls_id] = 1

    print(f"[INFO] Ultra-low classes (<10): {len(ultra_low)}, will use heavy augmentation instead of oversample")

    # 3. 处理每张图
    total_src, total_dst, aug_count = 0, 0, 0
    for stem, cls_ids in img_classes.items():
        max_mult = max(class_multiplier.get(c, 1) for c in cls_ids)
        has_ultra = bool(set(cls_ids) & ultra_low)

        src_lbl = labels_dir / f"{stem}.txt"
        src_img = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                src_img = candidate
                break
        if not src_img:
            continue

        total_src += 1
        original = cv2.imread(str(src_img))
        if original is None:
            continue

        # 始终保留一份原始
        shutil.copy2(src_img, output_img / f"{stem}{src_img.suffix}")
        shutil.copy2(src_lbl, output_lbl / f"{stem}.txt")
        total_dst += 1

        # 对需要多倍采样的图，复制
        for dup_idx in range(1, max_mult):
            dst_stem = f"{stem}_x{dup_idx}"
            shutil.copy2(src_img, output_img / f"{dst_stem}{src_img.suffix}")
            shutil.copy2(src_lbl, output_lbl / f"{dst_stem}.txt")
            total_dst += 1

        # 如果图里有极低频类，额外生成增强版本
        if has_ultra and max_mult == 1:
            for aug_idx in range(15):  # 每张极低频图生成 15 个增强版本
                augmented = augment_image(original, seed=hash(f"{stem}_{aug_idx}"))
                aug_stem = f"{stem}_aug{aug_idx}"
                cv2.imwrite(str(output_img / f"{aug_stem}.jpg"), augmented, [cv2.IMWRITE_JPEG_QUALITY, 95])
                shutil.copy2(src_lbl, output_lbl / f"{aug_stem}.txt")
                aug_count += 1
                total_dst += 1

    print(f"[INFO] {total_src} source → {total_dst} total ({aug_count} augmented variants for ultra-low classes)")

    # 4. 拷贝 val/test
    yolo_root = labels_dir.parent.parent
    for split in ["val", "test"]:
        src_img_dir = yolo_root / "images" / split
        src_lbl_dir = yolo_root / "labels" / split
        dst_img = output_dir / "images" / split
        dst_lbl = output_dir / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for f in src_img_dir.iterdir():
            shutil.copy2(f, dst_img / f.name)
        for f in src_lbl_dir.iterdir():
            shutil.copy2(f, dst_lbl / f.name)
        print(f"[INFO] Copied {split}: {sum(1 for _ in src_img_dir.iterdir())} images")

    # 5. dataset.yaml
    src_yaml = yolo_root / "dataset.yaml"
    with open(src_yaml) as f:
        ds_cfg = yaml.safe_load(f)
    ds_cfg["path"] = str(output_dir.absolute().as_posix())
    ds_cfg["train"] = str(output_img.absolute().as_posix())
    ds_cfg["val"] = str((output_dir / "images" / "val").absolute().as_posix())
    if (output_dir / "images" / "test").exists():
        ds_cfg["test"] = str((output_dir / "images" / "test").absolute().as_posix())
    with open(output_dir / "dataset.yaml", "w") as f:
        yaml.dump(ds_cfg, f)
    print(f"[INFO] dataset.yaml → {output_dir / 'dataset.yaml'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", default="data/processed/yolo_augmented")
    parser.add_argument("--threshold", type=int, default=200)
    args = parser.parse_args()
    oversample_with_augment(Path(args.images), Path(args.labels), Path(args.out), args.threshold)


if __name__ == "__main__":
    main()
