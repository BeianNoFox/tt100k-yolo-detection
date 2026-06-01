"""自定义早停 + Per-class AP 回调"""
import csv
from pathlib import Path


class RobustEarlyStopping:
    def __init__(self, patience=8, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.best_map = -1.0
        self.counter = 0

    def __call__(self, trainer):
        epoch = getattr(trainer, "epoch", 0)
        try:
            box = getattr(trainer.metrics, "box", None) if hasattr(trainer, "metrics") else None
            if box is None:
                return
            cur = float(box.map50) if box.map50 is not None else -1
        except Exception:
            return
        if cur < 0:
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
            trainer.model.ema = None  # trigger stop
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
        self._done = set()

    def __call__(self, trainer):
        epoch = getattr(trainer, "epoch", -1)
        if epoch in self._done:
            return
        self._done.add(epoch)
        try:
            box = getattr(trainer.metrics, "box", None) if hasattr(trainer, "metrics") else None
            if box is None:
                return
            ap_idx = getattr(box, "ap_class_index", None)
            ap50 = getattr(box, "ap50", None)
            if ap_idx is None or ap50 is None or len(ap_idx) == 0:
                return
            per_class = {}
            for idx, ap in zip(ap_idx.cpu().tolist(), ap50.cpu().tolist()):
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
        RobustEarlyStopping(patience=8, min_delta=0.001),
        PerClassAPCallback(output_dir=experiment_dir, class_names=class_names),
    ]
