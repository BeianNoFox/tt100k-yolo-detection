"""从实验日志自动生成详细 Markdown 报告（论文素材）"""
import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np


def generate_report(exp_dir: Path):
    config_path = exp_dir / "config.yaml"
    results_path = exp_dir / "results.csv"
    ap_path = exp_dir / "per_class_ap.csv"

    if not results_path.exists():
        print(f"[WARN] No results.csv in {exp_dir}, skipping")
        return

    config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip()

    # 取最后一轮 + 最优轮
    last = df.iloc[-1]
    best_idx = df["metrics/mAP50(B)"].idxmax() if "metrics/mAP50(B)" in df.columns else len(df) - 1
    best = df.iloc[best_idx]

    report_lines = []

    exp_name = config.get("experiment", exp_dir.name)
    report_lines.append(f"# {exp_name} — 实验报告\n")
    report_lines.append(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    report_lines.append("---\n")

    # 1. 实验配置
    report_lines.append("## 1. 实验配置\n")
    for key in ["model", "imgsz", "epochs", "patience", "batch", "lr0", "optimizer"]:
        if key in config:
            report_lines.append(f"- **{key}**: {config[key]}\n")

    tiling = config.get("tiling", {})
    report_lines.append(f"- **切片**: {'开' if tiling.get('enabled') else '关'}")
    if tiling.get("enabled"):
        report_lines.append(f" (overlap={tiling.get('overlap')})\n")
    else:
        report_lines.append("\n")

    aug = config.get("augment", {})
    report_lines.append(f"- **增强**: mosaic={aug.get('mosaic')}, mixup={aug.get('mixup')}, fliplr={aug.get('fliplr')}\n")

    lt = config.get("long_tail", {})
    report_lines.append(f"- **长尾处理**: oversample={'开' if lt.get('oversample') else '关'}, "
                        f"class_weight={'开' if lt.get('class_weight_loss') else '关'}\n\n")

    # 2. 训练概况
    report_lines.append("## 2. 训练概况\n")
    actual_epochs = len(df)
    report_lines.append(f"- 实际训练轮次: {actual_epochs} / {config.get('epochs', '?')} 配置\n")
    report_lines.append(f"- Patience 触发: {'是' if actual_epochs < config.get('epochs', 999) else '否'}\n")
    if "train/box_loss" in df.columns:
        report_lines.append(f"- 初始 train/box_loss: {df['train/box_loss'].iloc[0]:.4f}\n")
        report_lines.append(f"- 最终 train/box_loss: {df['train/box_loss'].iloc[-1]:.4f}\n")
    if "val/box_loss" in df.columns:
        report_lines.append(f"- 初始 val/box_loss: {df['val/box_loss'].iloc[0]:.4f}\n")
        report_lines.append(f"- 最终 val/box_loss: {df['val/box_loss'].iloc[-1]:.4f}\n")
    report_lines.append("\n")

    # 3. 核心指标
    report_lines.append("## 3. 核心指标\n")
    report_lines.append("| 指标 | 最优值 (epoch) | 最终值 |\n")
    report_lines.append("|------|--------------|--------|\n")

    metric_cols = [
        ("mAP@0.5", "metrics/mAP50(B)"),
        ("mAP@0.5:0.95", "metrics/mAP50-95(B)"),
        ("Precision", "metrics/precision(B)"),
        ("Recall", "metrics/recall(B)"),
    ]

    for name, col in metric_cols:
        if col in df.columns:
            best_val = df[col].max()
            best_ep = df[col].idxmax()
            last_val = df[col].iloc[-1]
            report_lines.append(f"| {name} | **{best_val:.4f}** (epoch {best_ep}) | {last_val:.4f} |\n")
    report_lines.append("\n")

    # 4. Per-class AP 分析
    if ap_path.exists():
        ap_df = pd.read_csv(ap_path)
        last_ap = ap_df.iloc[-1]
        ap_cols = [c for c in ap_df.columns if c.startswith("AP_")]
        ap_data = [(c.replace("AP_", ""), float(last_ap[c])) for c in ap_cols]
        ap_data.sort(key=lambda x: -x[1])

        report_lines.append("## 4. 各类别 AP 分析\n")
        report_lines.append(f"- 平均 AP: {np.mean([v for _, v in ap_data]):.4f}\n")
        report_lines.append(f"- 最高: {ap_data[0][0]} ({ap_data[0][1]:.4f})\n")
        report_lines.append(f"- 最低: {ap_data[-1][0]} ({ap_data[-1][1]:.4f})\n")
        report_lines.append(f"- AP<0.1 类别数: {sum(1 for _, v in ap_data if v < 0.1)}\n")
        report_lines.append(f"- AP<0.05 类别数: {sum(1 for _, v in ap_data if v < 0.05)}\n\n")

        report_lines.append("| 排名 | 类别 | AP@0.5 | 档位 |\n")
        report_lines.append("|------|------|--------|------|\n")
        for rank, (cls, ap) in enumerate(ap_data, 1):
            if ap >= 0.7:
                tier = "优"
            elif ap >= 0.5:
                tier = "良"
            elif ap >= 0.2:
                tier = "中"
            else:
                tier = "差"
            report_lines.append(f"| {rank} | {cls} | {ap:.4f} | {tier} |\n")
        report_lines.append("\n")

    # 5. 图表清单
    report_lines.append("## 5. 生成图表\n\n")
    plots_dir = exp_dir / "plots"
    if plots_dir.exists():
        for png in sorted(plots_dir.glob("*.png")):
            report_lines.append(f"- `{png.name}`\n")
    eval_dir = exp_dir / "eval"
    if eval_dir.exists():
        for png in sorted(eval_dir.glob("*.png")):
            report_lines.append(f"- `eval/{png.name}`\n")
    report_lines.append("\n")

    # 6. 观察与建议（占位 — 待人工/AI 填写）
    report_lines.append("## 6. 观察与建议\n\n")
    report_lines.append("> *待分析*\n\n")

    report_lines.append("---\n")
    report_lines.append("*报告由 src/eval/report.py 自动生成*\n")

    report_path = exp_dir / "report.md"
    report_path.write_text("".join(report_lines), encoding="utf-8")
    print(f"[OK] Report saved to {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument("--exp-dir", required=True, help="Path to experiment directory")
    args = parser.parse_args()
    generate_report(Path(args.exp_dir))


if __name__ == "__main__":
    main()
