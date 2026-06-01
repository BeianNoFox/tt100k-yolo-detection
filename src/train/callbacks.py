"""Ultralytics YOLOv8 回调：每 epoch 写入 per-class AP"""
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
        self._done = set()

    def __call__(self, trainer):
        epoch = getattr(trainer, "epoch", -1)
        if epoch in self._done:
            return
        self._done.add(epoch)

        try:
            # 方式 A: 直接读取 trainer.metrics (Ultralytics 8.4 在 val 后设置)
            metrics = getattr(trainer, "metrics", None)
            if metrics is None:
                return
            box = getattr(metrics, "box", None)
            if box is None:
                return

            ap_idx = getattr(box, "ap_class_index", None)
            ap50 = getattr(box, "ap50", None)

            if ap_idx is None or ap50 is None or len(ap_idx) == 0:
                return

            per_class = {}
            idx_list = ap_idx.cpu().tolist() if hasattr(ap_idx, "cpu") else list(ap_idx)
            ap_list = ap50.cpu().tolist() if hasattr(ap50, "cpu") else list(ap50)
            for idx, ap in zip(idx_list, ap_list):
                per_class[idx] = ap

            map50 = float(box.map50) if getattr(box, "map50", None) is not None else 0.0
            map_val = float(box.map) if getattr(box, "map", None) is not None else 0.0

            row = [epoch]
            for i in range(len(self.class_names)):
                row.append(round(per_class.get(i, 0.0), 4))
            row.append(round(map50, 6))
            row.append(round(map_val, 6))

            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

            print(f"[Callback] Epoch {epoch}: {len(per_class)} classes, mAP50={map50:.4f}")

        except Exception as e:
            # 方式 B: 从 trainer.validator 拿
            try:
                validator = getattr(trainer, "validator", None)
                if validator is None:
                    return
                metrics = getattr(validator, "metrics", None)
                if metrics is None:
                    return
                box = getattr(metrics, "box", None)
                if box is None:
                    return

                ap_idx = getattr(box, "ap_class_index", None)
                ap50 = getattr(box, "ap50", None)
                if ap_idx is None or ap50 is None or len(ap_idx) == 0:
                    return

                per_class = {}
                idx_list = ap_idx.cpu().tolist() if hasattr(ap_idx, "cpu") else list(ap_idx)
                ap_list = ap50.cpu().tolist() if hasattr(ap50, "cpu") else list(ap50)
                for idx, ap in zip(idx_list, ap_list):
                    per_class[idx] = ap

                map50 = float(box.map50) if getattr(box, "map50", None) is not None else 0.0
                map_val = float(box.map) if getattr(box, "map", None) is not None else 0.0

                row = [epoch]
                for i in range(len(self.class_names)):
                    row.append(round(per_class.get(i, 0.0), 4))
                row.append(round(map50, 6))
                row.append(round(map_val, 6))

                with open(self.csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(row)

                print(f"[Callback] Epoch {epoch}: {len(per_class)} classes, mAP50={map50:.4f} (via validator)")
            except Exception as e2:
                print(f"[Callback] Epoch {epoch}: API access failed — {e2}")


def make_callbacks(experiment_dir: Path, class_names: list):
    return [PerClassAPCallback(output_dir=experiment_dir, class_names=class_names)]
