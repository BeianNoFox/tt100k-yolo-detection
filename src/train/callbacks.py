"""Ultralytics YOLOv8 自定义回调：追踪每 epoch per-class AP"""
import csv
from pathlib import Path
import numpy as np


class PerClassAPCallback:
    def __init__(self, output_dir: Path, class_names: list):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        self.csv_path = self.output_dir / "per_class_ap.csv"
        header = ["epoch"] + [f"AP_{name}" for name in class_names] + ["mAP50", "mAP50_95"]
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    def __call__(self, trainer):
        epoch = getattr(trainer, "epoch", -1)
        try:
            validator = getattr(trainer, "validator", None)
            if validator is None:
                return
            metrics = getattr(validator, "metrics", None)
            if metrics is None:
                return

            maps = getattr(metrics, "maps", None)
            ap_idx = getattr(metrics, "ap_class_index", None)

            if maps is None or ap_idx is None or len(ap_idx) == 0:
                print(f"[Callback] Epoch {epoch}: no per-class data")
                return

            row = [epoch]
            for i in range(len(self.class_names)):
                matched = [j for j, idx in enumerate(ap_idx) if idx == i]
                row.append(round(float(maps[matched[0]][0]), 4) if matched else 0.0)

            rd = metrics.results_dict
            row.append(round(rd.get("metrics/mAP50(B)", 0.0), 6))
            row.append(round(rd.get("metrics/mAP50-95(B)", 0.0), 6))

            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            print(f"[Callback] Epoch {epoch}: {e}")


def make_callbacks(experiment_dir: Path, class_names: list):
    return [PerClassAPCallback(output_dir=experiment_dir, class_names=class_names)]
