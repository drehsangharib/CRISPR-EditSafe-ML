from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def permutation_importance_table(model_bundle, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance
    result = permutation_importance(
        model_bundle.model,
        X[model_bundle.feature_columns],
        y,
        n_repeats=n_repeats,
        random_state=42,
        scoring="average_precision",
    )
    return pd.DataFrame({
        "feature": model_bundle.feature_columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)


def plot_top_features(importance_df: pd.DataFrame, output: str | Path, top_n: int = 20) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    top = importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(4, top_n * 0.25)))
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Permutation importance")
    plt.ylabel("Feature")
    plt.title("Top model features")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
