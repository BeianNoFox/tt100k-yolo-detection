"""从 YAML 配置读取参数并运行 YOLOv8 训练实验，自动收集全量日志"""
import sys
import shutil
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO
from src.train.callbacks import make_callbacks


def run_experiment(config_path: Path, experiments_root: Path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["experiment"]
    exp_dir = experiments_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置副本
    shutil.copy2(config_path, exp_dir / "config.yaml")

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"Description: {cfg.get('description', 'N/A')}")
    print(f"Output: {exp_dir}")
    print(f"{'='*60}\n")

    # 加载模型
    model_path = cfg.get("model", "yolov8s.pt")
    model = YOLO(model_path)

    # 模型摘要
    model_info = str(model.model) if hasattr(model, "model") else "See args.yaml"
    (exp_dir / "model_summary.txt").write_text(model_info, encoding="utf-8")

    # 解析 data yaml 获取类别名
    data_path = cfg.get("data", "data/processed/yolo/dataset.yaml")
    with open(data_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg.get("names", [f"class_{i}" for i in range(100)])
    num_classes = data_cfg.get("nc", len(class_names))

    # 长尾处理说明
    lt = cfg.get("long_tail", {})
    if lt.get("oversample"):
        tiers = lt.get("oversample_multiplier", [3, 5, 10])
        threshold = lt.get("oversample_threshold", 200)
        print(f"[INFO] Oversampling enabled — threshold={threshold}, tiers={tiers}")
    if lt.get("class_weight_loss"):
        print("[INFO] Class weights applied via oversample multiplier tiers")

    # 注册回调
    callbacks = make_callbacks(exp_dir, class_names)

    # 构建训练参数
    train_args = {
        "data": data_path,
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
        "save_period": -1,
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

    # 开始训练
    results = model.train(**train_args)

    # 收集产物 → experiment 根目录
    run_dirs = sorted(exp_dir.glob("run*"))
    if run_dirs:
        run_dir = run_dirs[-1] if run_dirs[-1].is_dir() else run_dirs[0]
        # weights
        for pt in Path(run_dir).glob("weights/*.pt"):
            shutil.copy2(pt, exp_dir / pt.name)
        # results.csv
        results_csv = Path(run_dir) / "results.csv"
        if results_csv.exists():
            shutil.copy2(results_csv, exp_dir / "results.csv")
        # 自动生成的图片
        plots_out = exp_dir / "plots"
        plots_out.mkdir(exist_ok=True)
        for png in Path(run_dir).glob("*.png"):
            shutil.copy2(png, plots_out / png.name)
        # args.yaml
        args_yaml = Path(run_dir) / "args.yaml"
        if args_yaml.exists():
            shutil.copy2(args_yaml, exp_dir / "args.yaml")

    print(f"\n[OK] '{exp_name}' complete → {exp_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run a YOLOv8 training experiment")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--experiments-root", default="experiments", help="Root directory")
    args = parser.parse_args()

    run_experiment(Path(args.config), Path(args.experiments_root))


if __name__ == "__main__":
    main()
