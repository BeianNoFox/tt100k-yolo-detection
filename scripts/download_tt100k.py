"""TT100K 数据下载指引脚本"""
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

4. 解压后 data/raw/tt100k_2021/ 下应该有:
   - annotations_all.json (标注文件, ~3.5MB)
   - train/                (训练图片, 2048x2048 jpg)
   - test/                 (测试图片)
   - other/                (无标注图片, 丢弃)

5. 确认后运行预处理流水线:
   python src/preprocess/class_stats.py --anno data/raw/tt100k_2021/annotations_all.json
   python src/preprocess/filter.py --anno data/raw/tt100k_2021/annotations_all.json --min-instances 100
   python src/preprocess/convert.py --anno data/processed/filtered/annotations_filtered.json --images data/raw/tt100k_2021
   python src/preprocess/split.py --labels data/processed/yolo/labels --images data/raw/tt100k_2021
========================================
"""


def main():
    raw_dir = Path("data/raw/tt100k_2021")
    raw_dir.mkdir(parents=True, exist_ok=True)

    anno_path = raw_dir / "annotations_all.json"
    if anno_path.exists():
        print("[OK] annotations_all.json already exists")
        print("Dataset ready. Run preprocess scripts directly.")
    else:
        print(DOWNLOAD_INSTRUCTIONS)


if __name__ == "__main__":
    main()
