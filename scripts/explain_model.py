#!/usr/bin/env python
from __future__ import annotations

import argparse
import pandas as pd
from crispr_editsafe.features import feature_columns
from crispr_editsafe.models import load_model
from crispr_editsafe.explain import permutation_importance_table, plot_top_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate permutation importance table and plot")
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--importance-out", required=True)
    parser.add_argument("--plot-out", required=True)
    args = parser.parse_args()
    bundle = load_model(args.model)
    df = pd.read_csv(args.features)
    importance = permutation_importance_table(bundle, df[feature_columns(df)], df["label"])
    importance.to_csv(args.importance_out, index=False)
    plot_top_features(importance, args.plot_out)
    print(f"Wrote importance: {args.importance_out}")


if __name__ == "__main__":
    main()
