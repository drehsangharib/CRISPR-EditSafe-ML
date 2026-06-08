#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
from crispr_editsafe.evaluate import classification_metrics, make_train_test_split, write_metrics
from crispr_editsafe.features import feature_columns
from crispr_editsafe.models import fit_model, predict_proba, save_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CRISPR off-target prediction model")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="configs/train_config.json")
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--predictions-out", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    df = pd.read_csv(args.input)
    train_df, test_df, split_strategy = make_train_test_split(
        df,
        test_size=config.get("test_size", 0.25),
        random_state=config.get("random_state", 42),
    )
    cols = feature_columns(df)
    bundle = fit_model(train_df[cols], train_df["label"], model_name=config.get("model_name", "logistic_regression"))
    y_score = predict_proba(bundle, test_df[cols])
    metrics = classification_metrics(test_df["label"], y_score, threshold=config.get("threshold", 0.5))
    metrics["split_strategy"] = split_strategy
    metrics["model_name"] = bundle.model_name
    metrics["n_features"] = len(cols)

    pred_df = test_df[[c for c in ["sample_id", "source", "sgRNA", "target", "label"] if c in test_df.columns]].copy()
    pred_df["off_target_risk"] = y_score
    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.predictions_out, index=False)
    save_model(bundle, args.model_out)
    write_metrics(metrics, args.metrics_out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
