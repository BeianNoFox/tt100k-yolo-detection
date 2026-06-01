"""Ultralytics YOLOv8 自定义回调：追踪每 epoch per-class AP"""
import csv
from pathlib import Path


class PerClassAPCallback:
    """在每轮验证结束后计算并追加写入 per-class AP"""

    def __init__(self, output_dir: Path, class_names: list):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names
        self.csv_path = self.output_dir / "per_class_ap.csv"
        self._init_csv()

    def _init_csv(self):
        header = ["epoch"] + [f"AP_{name}" for name in self.class_names] + ["mAP50", "mAP50_95"]
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"[Callback] Per-class AP CSV initialized: {self.csv_path}")

    def __call__(self, trainer):
        """Ultralytics 在每轮验证后调用此回调"""
        epoch = getattr(trainer, "epoch", -1)

        try:
            validator = getattr(trainer, "validator", None)
            if validator is None:
                return

            metrics = getattr(validator, "metrics", None)
            if metrics is None:
                return

            # Ultralytics metrics.box 包含 ap_class_index 和各阈值AP
            box_metrics = getattr(metrics, "box", None)
            if box_metrics is None:
                return

            per_class_ap = {}
            if hasattr(box_metrics, "ap_class_index") and box_metrics.ap_class_index is not None:
                ap50 = box_metrics.ap50
                for idx, ap_val in zip(box_metrics.ap_class_index.cpu().numpy(),
                                       ap50.cpu().numpy()):
                    if idx < len(self.class_names):
                        per_class_ap[int(idx)] = float(ap_val)

            # 写入一行
            row = [epoch]
            for i in range(len(self.class_names)):
                row.append(round(per_class_ap.get(i, 0.0), 6))
            row.append(round(float(box_metrics.map50), 6) if box_metrics.map50 is not None else 0.0)
            row.append(round(float(box_metrics.map), 6) if box_metrics.map is not None else 0.0)

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

        except Exception as e:
            print(f"[Callback] Warning: epoch {epoch} per-class AP collection failed: {e}")


def make_callbacks(experiment_dir: Path, class_names: list):
    """创建实验所需的全部回调"""
    return [PerClassAPCallback(output_dir=experiment_dir, class_names=class_names)]
