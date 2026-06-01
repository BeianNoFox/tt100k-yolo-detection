"""Ultralytics YOLOv8 自定义回调：追踪每 epoch per-class AP"""
import csv
from pathlib import Path


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

            # Ultralytics 8.4: DetMetrics.box = Metric 对象
            box = getattr(metrics, "box", None)
            if box is None:
                return

            ap_idx = getattr(box, "ap_class_index", None)
            ap50 = getattr(box, "ap50", None)

            if ap_idx is None or ap50 is None:
                return
            if len(ap_idx) == 0:
                return

            # ap_idx: tensor of class indices, ap50: tensor of AP values
            ap_dict = {}
            for idx, ap in zip(ap_idx.cpu().tolist(), ap50.cpu().tolist()):
                ap_dict[idx] = ap

            map50 = float(box.map50) if getattr(box, "map50", None) is not None else 0.0
            map_val = float(box.map) if getattr(box, "map", None) is not None else 0.0

            row = [epoch]
            for i in range(len(self.class_names)):
                row.append(round(ap_dict.get(i, 0.0), 4))
            row.append(round(map50, 6))
            row.append(round(map_val, 6))

            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

        except Exception as e:
            print(f"[Callback] Epoch {epoch}: {e}")


def make_callbacks(experiment_dir: Path, class_names: list):
    return [PerClassAPCallback(output_dir=experiment_dir, class_names=class_names)]
