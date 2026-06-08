#!/usr/bin/env python
from __future__ import annotations

import argparse
from crispr_editsafe.io import load_inference_data, write_csv
from crispr_editsafe.features import featurize_dataframe
from crispr_editsafe.models import load_model, predict_proba


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict CRISPR off-target risk for candidate pairs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bundle = load_model(args.model)
    df = load_inference_data(args.input)
    feat = featurize_dataframe(df)
    df["off_target_risk"] = predict_proba(bundle, feat)
    df = df.sort_values("off_target_risk", ascending=False)
    write_csv(df, args.output)
    print(f"Wrote predictions: {args.output}")


if __name__ == "__main__":
    main()
