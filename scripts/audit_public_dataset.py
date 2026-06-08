#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd

from crispr_editsafe.public_data import read_table, infer_mapping, clean_sequence


def safe_value_counts(series: pd.Series, max_items: int = 20) -> dict:
    counts = series.value_counts(dropna=False).head(max_items)
    return {str(k): int(v) for k, v in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a public CRISPR off-target table and suggest adapter settings.")
    parser.add_argument("--input", required=True, help="CSV/TSV/XLSX public dataset file")
    parser.add_argument("--output", default=None, help="Optional JSON audit report path")
    parser.add_argument("--source-name", default="public_dataset", help="Source name for suggested command")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Score threshold for suggested command")
    args = parser.parse_args()

    df = read_table(args.input)
    report: dict = {
        "input": args.input,
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": list(map(str, df.columns)),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "head_records": df.head(5).astype(str).to_dict(orient="records"),
    }

    try:
        mapping = infer_mapping(df)
        report["inferred_mapping"] = mapping.__dict__
        guide_clean = df[mapping.guide_col].map(clean_sequence)
        target_clean = df[mapping.target_col].map(clean_sequence)
        report["guide_length_counts_after_cleaning"] = safe_value_counts(guide_clean.str.len())
        report["target_length_counts_after_cleaning"] = safe_value_counts(target_clean.str.len())
        report["unique_guides_after_cleaning"] = int(guide_clean.nunique())

        command_parts = [
            "python scripts/prepare_public_dataset.py",
            f"--input {args.input}",
            f"--output data/processed/{args.source_name}_standardized.csv",
            f"--source-name {args.source_name}",
            f"--guide-col \"{mapping.guide_col}\"",
            f"--target-col \"{mapping.target_col}\"",
        ]
        if mapping.label_col:
            report["label_value_counts"] = safe_value_counts(df[mapping.label_col])
            command_parts.append(f"--label-col \"{mapping.label_col}\"")
        elif mapping.score_col:
            report["score_summary"] = pd.to_numeric(df[mapping.score_col], errors="coerce").describe().to_dict()
            command_parts.append(f"--score-col \"{mapping.score_col}\"")
            command_parts.append(f"--score-threshold {args.score_threshold}")
        else:
            report["warning"] = "Guide and target columns were inferred, but no label or score column was inferred."
        report["suggested_prepare_command"] = " ".join(command_parts)
    except Exception as exc:
        report["mapping_error"] = str(exc)

    print(json.dumps(report, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote audit report: {out}")


if __name__ == "__main__":
    main()
