from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def make_train_test_split(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    if "sgRNA" in df.columns and df["sgRNA"].nunique() >= 4:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(df, groups=df["sgRNA"]))
        return df.iloc[train_idx].copy(), df.iloc[test_idx].copy(), "sgRNA_disjoint"
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"] if df["label"].nunique() == 2 else None
    )
    return train_df.copy(), test_df.copy(), "row_random"


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auprc": None,
        "auroc": None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def write_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
