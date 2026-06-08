#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def save_metric_barplot(summary: pd.DataFrame, output: Path) -> None:
    metrics = [m for m in ["auprc", "auroc", "f1", "precision", "recall"] if m in summary.columns]
    if not metrics:
        return
    plot_df = summary.set_index("model_name")[metrics]
    ax = plot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("CRISPR off-target model benchmark metrics")
    ax.legend(loc="lower right")
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()


def save_prediction_distribution(predictions_dir: Path, output: Path) -> None:
    files = sorted(predictions_dir.glob("*_predictions.csv"))
    if not files:
        return
    plt.figure(figsize=(8, 5))
    for f in files:
        df = pd.read_csv(f)
        if "off_target_risk" not in df.columns:
            continue
        label = f.name.replace("_predictions.csv", "")
        plt.hist(df["off_target_risk"], bins=20, alpha=0.5, label=label)
    plt.xlabel("Predicted off-target risk")
    plt.ylabel("Candidate count")
    plt.title("Predicted risk score distributions")
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()


def generate_markdown(summary: pd.DataFrame, benchmark_dir: Path, output: Path, metric_plot: Path, risk_plot: Path) -> None:
    best_by_auprc = None
    if "auprc" in summary.columns and not summary.empty:
        best_by_auprc = summary.sort_values("auprc", ascending=False).iloc[0]

    lines = [
        "# CRISPR-EditSafe-ML Benchmark Report",
        "",
        "## Overview",
        "This report summarizes baseline model performance for a standardized CRISPR off-target dataset.",
        "",
        "## Benchmark summary",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    if best_by_auprc is not None:
        lines.extend([
            "## Best model by AUPRC",
            "",
            f"- Model: `{best_by_auprc['model_name']}`",
            f"- AUPRC: `{best_by_auprc.get('auprc')}`",
            f"- AUROC: `{best_by_auprc.get('auroc')}`",
            f"- F1: `{best_by_auprc.get('f1')}`",
            "",
        ])
    if metric_plot.exists():
        rel = metric_plot.relative_to(output.parent).as_posix() if metric_plot.is_relative_to(output.parent) else metric_plot.as_posix()
        lines.extend(["## Metric comparison", "", f"![Metric comparison]({rel})", ""])
    if risk_plot.exists():
        rel = risk_plot.relative_to(output.parent).as_posix() if risk_plot.is_relative_to(output.parent) else risk_plot.as_posix()
        lines.extend(["## Risk-score distributions", "", f"![Risk distributions]({rel})", ""])
    lines.extend([
        "## Interpretation notes",
        "",
        "- AUPRC is emphasized because CRISPR off-target data are often class-imbalanced.",
        "- sgRNA-disjoint evaluation is preferred when enough unique guides are available.",
        "- Small demo datasets can produce unstable or perfect-looking scores; use public benchmark datasets for meaningful interpretation.",
        "",
        "## Reproducibility",
        "",
        f"Benchmark directory: `{benchmark_dir}`",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures and Markdown report from benchmark outputs.")
    parser.add_argument("--benchmark-dir", required=True, help="Directory produced by scripts/benchmark_models.py")
    parser.add_argument("--output", default="reports/benchmark_report.md", help="Output Markdown report path")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    summary_path = benchmark_dir / "benchmark_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing benchmark summary: {summary_path}")
    summary = pd.read_csv(summary_path)

    output = Path(args.output)
    fig_dir = output.parent / "figures"
    metric_plot = fig_dir / "benchmark_metrics.png"
    risk_plot = fig_dir / "risk_score_distributions.png"
    save_metric_barplot(summary, metric_plot)
    save_prediction_distribution(benchmark_dir / "predictions", risk_plot)
    generate_markdown(summary, benchmark_dir, output, metric_plot, risk_plot)
    print(f"Wrote benchmark report: {output}")
    print(f"Wrote figure: {metric_plot}")
    print(f"Wrote figure: {risk_plot}")


if __name__ == "__main__":
    main()
