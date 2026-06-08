#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from crispr_editsafe.deep_models import encode_dataframe_pairs, train_cnn, predict_cnn_proba
from crispr_editsafe.evaluate import make_train_test_split, classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch 1D CNN on standardized CRISPR off-target pairs.")
    parser.add_argument("--input", required=True, help="Standardized CSV with sgRNA,target,label columns.")
    parser.add_argument("--output-dir", required=True, help="Output directory for CNN model, metrics, predictions, and training history.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="cpu", help="Use cpu by default. Use cuda only if PyTorch CUDA is installed.")
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is not installed. Install it with: pip install torch") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    train_df, test_df, split_strategy = make_train_test_split(df, test_size=0.25, random_state=42)
    train_inner_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df["label"] if train_df["label"].nunique() == 2 else None,
    )

    X_train = encode_dataframe_pairs(train_inner_df)
    y_train = train_inner_df["label"].values
    X_val = encode_dataframe_pairs(val_df)
    y_val = val_df["label"].values
    X_test = encode_dataframe_pairs(test_df)

    model, result = train_cnn(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        device=args.device,
    )

    y_score = predict_cnn_proba(model, X_test, device=args.device)
    metrics = classification_metrics(test_df["label"], y_score)
    metrics["model_name"] = "pytorch_cnn"
    metrics["split_strategy"] = split_strategy
    metrics["n_train"] = int(len(train_inner_df))
    metrics["n_validation"] = int(len(val_df))
    metrics["n_test"] = int(len(test_df))
    metrics["best_validation_loss"] = float(result.best_validation_loss)

    pred_df = test_df[[c for c in ["sample_id", "source", "sgRNA", "target", "label"] if c in test_df.columns]].copy()
    pred_df["off_target_risk"] = y_score

    torch.save(model.state_dict(), output_dir / "pytorch_cnn_state_dict.pt")
    pred_df.to_csv(output_dir / "pytorch_cnn_predictions.csv", index=False)
    (output_dir / "pytorch_cnn_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(result.history).to_csv(output_dir / "pytorch_cnn_training_history.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote CNN outputs to: {output_dir}")


if __name__ == "__main__":
    main()
