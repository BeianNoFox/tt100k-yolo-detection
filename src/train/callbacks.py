"""自定义早停 + Per-class AP 回调"""
import csv
from pathlib import Path
from collections import deque


def _to_list(obj):
    if hasattr(obj, "cpu"):
        return obj.cpu().tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return list(obj)


class RobustEarlyStopping:
    """
    基于移动均线的早停：
    - 每轮记录 mAP，取最近 `window` 轮的中位数/均值
    - 若近期均值不再高于已记录的峰值（margin > min_delta），计数 +1
    - 连续 patience 轮无实质提升 → 自动停止
    """
    def __init__(self, patience=5, min_delta=0.005, window=5):
        self.patience = patience          # 停滞 patience 轮后停
        self.min_delta = min_delta        # 提升阈值（~0.5 mAP 点）
        self.window = window              # 平滑窗口
        self.history = deque(maxlen=window)
        self.peak = -1.0                  # 已确认的最高 mAP
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
        if cur < 0.01:
            return

        self.history.append(cur)
        smoothed = sum(self.history) / len(self.history)  # 窗口均值

        if not self._started:
            self.peak = smoothed
            self._started = True
            print(f"[EarlyStop] Epoch {epoch}: baseline smoothed mAP={smoothed:.4f}")
            return

        if smoothed > self.peak + self.min_delta:
            self.peak = smoothed
            self.counter = 0
            print(f"[EarlyStop] Epoch {epoch}: smoothed mAP ↑ {smoothed:.4f} (peak={self.peak:.4f})")
        else:
            self.counter += 1
            direction = "↑" if smoothed > self.peak else "↓"
            print(f"[EarlyStop] Epoch {epoch}: smoothed mAP={smoothed:.4f} {direction}, plateau {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            print(f"[EarlyStop] Stopping — no meaningful improvement for {self.patience} epochs")
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
        RobustEarlyStopping(patience=5, min_delta=0.005, window=5),
        PerClassAPCallback(output_dir=experiment_dir, class_names=class_names),
    ]
