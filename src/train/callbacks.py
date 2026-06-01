"""自定义早停 + Per-class AP 回调"""
import csv
from pathlib import Path


def _to_list(obj):
    """兼容 torch tensor / numpy / list"""
    if hasattr(obj, "cpu"):
        return obj.cpu().tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return list(obj)


class RobustEarlyStopping:
    def __init__(self, patience=8, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.best_map = -1.0
        self.counter = 0
        self._started = False

    def __call__(self, trainer):
        epoch = getattr(trainer, "epoch", 0)
        try:
            box = getattr(trainer.metrics, "box", None) if hasattr(trainer, "metrics") else None
            if box is None:
                return
            cur = float(box.map50) if box.map50 is not None else -1
        except Exception:
            return
        if cur < 0.01:  # 不合理值跳过
            return

        # 首次调用：设基准
        if not self._started:
            self.best_map = cur
            self._started = True
            print(f"[EarlyStop] Epoch {epoch}: baseline mAP={cur:.4f}")
            return

        improvement = cur - self.best_map
        if improvement > self.min_delta:
            self.best_map = cur
            self.counter = 0
            print(f"[EarlyStop] Epoch {epoch}: mAP improved → {cur:.4f} (+{improvement:.4f})")
        else:
            self.counter += 1
            print(f"[EarlyStop] Epoch {epoch}: mAP={cur:.4f} (Δ={improvement:.4f}), stale {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            print(f"[EarlyStop] Auto-stopping at epoch {epoch}")
            if hasattr(trainer, "stopper"):
                trainer.stopper.stop = True
            trainer.stop_training = True


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
        epoch = getattr(trainer, "epoch", 0)
        try:
            box = getattr(trainer.metrics, "box", None) if hasattr(trainer, "metrics") else None
            if box is None:
                return
            ap_idx = getattr(box, "ap_class_index", None)
            ap50 = getattr(box, "ap50", None)
            if ap_idx is None or ap50 is None or len(ap_idx) == 0:
                return
            idx_list = _to_list(ap_idx)
            ap_list = _to_list(ap50)
            per_class = {}
            for idx, ap in zip(idx_list, ap_list):
                per_class[idx] = ap
            map50 = float(box.map50) if box.map50 is not None else 0.0
            map_val = float(box.map) if box.map is not None else 0.0
            row = [epoch]
            for i in range(len(self.class_names)):
                row.append(round(per_class.get(i, 0.0), 4))
            row.append(round(map50, 6))
            row.append(round(map_val, 6))
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(f"[Callback] Epoch {epoch}: {len(per_class)} classes, mAP50={map50:.4f}")
        except Exception as e:
            print(f"[Callback] Epoch {epoch}: {e}")


def make_callbacks(experiment_dir: Path, class_names: list):
    return [
        RobustEarlyStopping(patience=8, min_delta=0.005),
        PerClassAPCallback(output_dir=experiment_dir, class_names=class_names),
    ]
