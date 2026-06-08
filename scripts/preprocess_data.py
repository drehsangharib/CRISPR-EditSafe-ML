#!/usr/bin/env python
from __future__ import annotations

import argparse
from crispr_editsafe.io import load_training_data, write_csv
from crispr_editsafe.features import featurize_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Featurize CRISPR off-target pairs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = load_training_data(args.input)
    features = featurize_dataframe(df)
    write_csv(features, args.output)
    print(f"Wrote features: {args.output} shape={features.shape}")


if __name__ == "__main__":
    main()
