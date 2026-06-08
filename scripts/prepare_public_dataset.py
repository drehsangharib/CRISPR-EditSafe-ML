#!/usr/bin/env python
from __future__ import annotations

import argparse
from crispr_editsafe.public_data import read_table, standardize_public_dataset, write_standardized


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize a public CRISPR off-target CSV/TSV/XLSX file into CRISPR-EditSafe-ML schema.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-name", default="public_dataset")
    parser.add_argument("--guide-col", default=None)
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--score-col", default=None)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    args = parser.parse_args()

    raw = read_table(args.input)
    standardized = standardize_public_dataset(
        raw,
        source_name=args.source_name,
        guide_col=args.guide_col,
        target_col=args.target_col,
        label_col=args.label_col,
        score_col=args.score_col,
        score_threshold=args.score_threshold,
    )
    write_standardized(standardized, args.output)
    print(f"Input rows: {len(raw)}")
    print(f"Standardized rows: {len(standardized)}")
    print(f"Positive labels: {int(standardized['label'].sum())}")
    print(f"Unique guides: {standardized['sgRNA'].nunique()}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
