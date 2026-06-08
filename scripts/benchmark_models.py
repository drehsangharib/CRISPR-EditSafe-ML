#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from crispr_editsafe.features import featurize_dataframe, feature_columns
from crispr_editsafe.evaluate import make_train_test_split, classification_metrics
from crispr_editsafe.models import fit_model, predict_proba, save_model


def benchmark_one_model(features: pd.DataFrame, model_name: str, output_dir: Path) -> dict:
    train_df, test_df, split_strategy = make_train_test_split(features, test_size=0.25, random_state=42)
    cols = feature_columns(features)
    bundle = fit_model(train_df[cols], train_df["label"], model_name=model_name)
    scores = predict_proba(bundle, test_df[cols])
    metrics = classification_metrics(test_df["label"], scores)
    metrics["model_name"] = model_name
    metrics["split_strategy"] = split_strategy
    metrics["n_features"] = len(cols)
    metrics["n_train"] = int(len(train_df))
    metrics["n_test"] = int(len(test_df))

    model_dir = output_dir / "models"
    pred_dir = output_dir / "predictions"
    metric_dir = output_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    save_model(bundle, str(model_dir / f"{model_name}.joblib"))
    pred_df = test_df[[c for c in ["sample_id", "source", "sgRNA", "target", "label"] if c in test_df.columns]].copy()
    pred_df["off_target_risk"] = scores
    pred_df.to_csv(pred_dir / f"{model_name}_predictions.csv", index=False)
    (metric_dir / f"{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline models on a standardized CRISPR off-target dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "random_forest"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    features = featurize_dataframe(df)
    features.to_csv(output_dir / "features.csv", index=False)

    all_metrics = [benchmark_one_model(features, model_name=m, output_dir=output_dir) for m in args.models]
    summary = pd.DataFrame(all_metrics)
    summary.to_csv(output_dir / "benchmark_summary.csv", index=False)
    (output_dir / "benchmark_summary.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(summary[["model_name", "split_strategy", "n_train", "n_test", "auprc", "auroc", "f1", "precision", "recall"]].to_string(index=False))
    print(f"Wrote benchmark outputs to: {output_dir}")


if __name__ == "__main__":
    main()
