"""从实验日志生成论文级高质量图表（300dpi, 中文字体）"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def read_experiments(experiments_root: Path, exp_names: list):
    dfs = {}
    for name in exp_names:
        csv_path = experiments_root / name / "results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            dfs[name] = df
        else:
            print(f"[WARN] {csv_path} not found, skipping")
    return dfs


def plot_loss_curves(exp_name: str, df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    loss_pairs = [
        ("train/box_loss", "val/box_loss", "Box Loss"),
        ("train/cls_loss", "val/cls_loss", "Cls Loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
    ]
    for ax, (train_col, val_col, title) in zip(axes, loss_pairs):
        if train_col in df.columns:
            ax.plot(df.index, df[train_col], label="Train", color=COLORS[0], linewidth=1.5)
        if val_col in df.columns:
            ax.plot(df.index, df[val_col], label="Val", color=COLORS[1], linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{exp_name} — Loss Curves")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "loss_curves.png")
    plt.close(fig)


def plot_map_curve(exp_name: str, df: pd.DataFrame, output_dir: Path):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    if "metrics/mAP50(B)" in df.columns:
        ax1.plot(df.index, df["metrics/mAP50(B)"], color=COLORS[0], label="mAP@0.5", linewidth=2)
    if "metrics/mAP50-95(B)" in df.columns:
        ax2.plot(df.index, df["metrics/mAP50-95(B)"], color=COLORS[1], label="mAP@0.5:0.95", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("mAP@0.5", color=COLORS[0])
    ax2.set_ylabel("mAP@0.5:0.95", color=COLORS[1])
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax1.set_title(f"{exp_name} — mAP Curves")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "map_curve.png")
    plt.close(fig)


def plot_multi_exp_loss_compare(dfs: dict, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # 取所有实验的中位 epoch 数作为 X 轴上限
    max_epochs = min(max(len(df) for df in dfs.values()), 50)
    for metric, ax, title in [("val/box_loss", axes[0], "Val Box Loss"),
                               ("val/cls_loss", axes[1], "Val Cls Loss")]:
        for i, (exp_name, df) in enumerate(dfs.items()):
            if metric in df.columns:
                subset = df.iloc[:max_epochs]
                ax.plot(range(len(subset)), df[metric].iloc[:max_epochs],
                        label=exp_name, color=COLORS[i % len(COLORS)], linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Multi-Experiment Loss Comparison (first 50 epochs)")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "multi_exp_loss_compare.png")
    plt.close(fig)


def plot_multi_exp_map_bars(dfs: dict, output_dir: Path):
    exp_names = list(dfs.keys())
    map50_vals, map50_95_vals = [], []
    for name in exp_names:
        df = dfs[name]
        map50_vals.append(df["metrics/mAP50(B)"].max() if "metrics/mAP50(B)" in df.columns else 0)
        map50_95_vals.append(df["metrics/mAP50-95(B)"].max() if "metrics/mAP50-95(B)" in df.columns else 0)
    x = np.arange(len(exp_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, map50_vals, width, label="mAP@0.5", color=COLORS[0])
    bars2 = ax.bar(x + width/2, map50_95_vals, width, label="mAP@0.5:0.95", color=COLORS[1])
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("mAP")
    ax.set_title("Ablation Experiment — mAP Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "multi_exp_map_bars.png")
    plt.close(fig)


def plot_per_class_ap(class_ap_csv: Path, output_dir: Path):
    if not class_ap_csv.exists():
        print(f"[WARN] {class_ap_csv} not found")
        return
    df = pd.read_csv(class_ap_csv)
    if len(df) == 0:
        print(f"[WARN] {class_ap_csv} is empty, skipping")
        return
    last_row = df.iloc[-1]
    ap_cols = [c for c in df.columns if c.startswith("AP_")]
    ap_values = [last_row[c] for c in ap_cols]
    class_labels = [c.replace("AP_", "") for c in ap_cols]
    sorted_idx = np.argsort(ap_values)[::-1]
    sorted_labels = [class_labels[i] for i in sorted_idx]
    sorted_vals = [ap_values[i] for i in sorted_idx]
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = [COLORS[0] if v >= 0.5 else COLORS[1] for v in sorted_vals]
    ax.bar(range(len(sorted_labels)), sorted_vals, color=colors, width=0.7)
    ax.set_xticks(range(len(sorted_labels)))
    ax.set_xticklabels(sorted_labels, rotation=90, fontsize=7)
    ax.set_ylabel("AP@0.5")
    ax.set_title("Per-Class AP (Final Epoch)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "per_class_ap_bar.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate thesis-quality plots from experiment logs")
    parser.add_argument("--experiments-root", default="experiments")
    parser.add_argument("--exps", nargs="+",
                        default=["exp1_baseline", "exp2_tiling_o20", "exp3_longtail", "exp4_finetune"])
    parser.add_argument("--out", default="experiments/comparison")
    args = parser.parse_args()

    root = Path(args.experiments_root)
    dfs = read_experiments(root, args.exps)
    if not dfs:
        print("[ERROR] No experiment results found. Run experiments first.")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for exp_name, df in dfs.items():
        exp_plots_dir = root / exp_name / "plots"
        plot_loss_curves(exp_name, df, exp_plots_dir)
        plot_map_curve(exp_name, df, exp_plots_dir)
        print(f"[OK] Single-experiment plots for {exp_name}")

    plot_multi_exp_loss_compare(dfs, out)
    plot_multi_exp_map_bars(dfs, out)
    print(f"[OK] Multi-experiment comparison in {out}")

    for exp_name in args.exps:
        ap_csv = root / exp_name / "per_class_ap.csv"
        if ap_csv.exists():
            plot_per_class_ap(ap_csv, root / exp_name / "eval")
            print(f"[OK] Per-class AP for {exp_name}")


if __name__ == "__main__":
    main()
