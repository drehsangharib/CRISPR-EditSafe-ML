from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re
import pandas as pd

DNA_ALLOWED = re.compile(r"[^ACGTUNacgtun_\-\.]")

GUIDE_CANDIDATES = [
    "sgrna", "sgRNA", "grna", "gRNA", "guide", "guide_seq", "guide_sequence",
    "spacer", "protospacer", "on_seq", "on_target", "ontarget", "on-target",
    "on_target_sequence", "target_sequence", "targetsite", "target_site"
]

TARGET_CANDIDATES = [
    "target", "dna", "dna_seq", "dna_sequence", "offtarget", "off_target", "off-target",
    "off_target_sequence", "offtarget_sequence", "off_seq", "off-target_sequence",
    "candidate", "candidate_sequence", "genomic_sequence", "site_sequence"
]

LABEL_CANDIDATES = [
    "label", "y", "class", "active", "is_active", "validated", "is_validated",
    "true_offtarget", "true_off_target", "observed", "detected", "activity_binary"
]

SCORE_CANDIDATES = [
    "score", "activity", "indel", "indel_rate", "read_count", "reads", "crispr_net_score",
    "CRISPR_Net_score", "cleavage_score", "editing_rate", "validated_score"
]


@dataclass
class ColumnMapping:
    guide_col: str
    target_col: str
    label_col: str | None = None
    score_col: str | None = None


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    col_list = list(columns)
    norm_to_original = {_norm_col(c): c for c in col_list}
    for candidate in candidates:
        key = _norm_col(candidate)
        if key in norm_to_original:
            return norm_to_original[key]
    for original in col_list:
        norm = _norm_col(original)
        for candidate in candidates:
            if _norm_col(candidate) in norm:
                return original
    return None


def clean_sequence(seq: object, keep_length: int = 20) -> str:
    """Clean guide/target sequences and return first keep_length nucleotides."""
    if pd.isna(seq):
        return ""
    s = str(seq).strip().upper().replace("U", "T")
    s = s.replace("_", "").replace("-", "").replace(".", "")
    s = DNA_ALLOWED.sub("", s)
    return s[:keep_length]


def infer_mapping(
    df: pd.DataFrame,
    guide_col: str | None = None,
    target_col: str | None = None,
    label_col: str | None = None,
    score_col: str | None = None,
) -> ColumnMapping:
    guide = guide_col or find_column(df.columns, GUIDE_CANDIDATES)
    target = target_col or find_column(df.columns, TARGET_CANDIDATES)
    label = label_col or find_column(df.columns, LABEL_CANDIDATES)
    score = score_col or find_column(df.columns, SCORE_CANDIDATES)

    if guide is None:
        raise ValueError(
            "Could not infer guide column. Pass --guide-col explicitly. "
            f"Available columns: {list(df.columns)}"
        )
    if target is None:
        raise ValueError(
            "Could not infer target/off-target column. Pass --target-col explicitly. "
            f"Available columns: {list(df.columns)}"
        )
    return ColumnMapping(guide_col=guide, target_col=target, label_col=label, score_col=score)


def _label_from_value(value: object, positive_values: set[str], negative_values: set[str]) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in positive_values:
        return 1
    if text in negative_values:
        return 0
    try:
        number = float(text)
    except ValueError:
        return None
    if number in (0.0, 1.0):
        return int(number)
    return None


def standardize_public_dataset(
    df: pd.DataFrame,
    source_name: str = "public_dataset",
    guide_col: str | None = None,
    target_col: str | None = None,
    label_col: str | None = None,
    score_col: str | None = None,
    score_threshold: float = 0.0,
    positive_values: Iterable[str] = ("1", "true", "yes", "active", "validated", "positive", "pos"),
    negative_values: Iterable[str] = ("0", "false", "no", "inactive", "not_validated", "negative", "neg"),
    min_length: int = 20,
) -> pd.DataFrame:
    """Convert a public CRISPR off-target table to sample_id,source,sgRNA,target,label."""
    mapping = infer_mapping(df, guide_col=guide_col, target_col=target_col, label_col=label_col, score_col=score_col)
    positives = {str(x).strip().lower() for x in positive_values}
    negatives = {str(x).strip().lower() for x in negative_values}

    out = pd.DataFrame()
    out["sample_id"] = [f"{source_name}_{i}" for i in range(len(df))]
    out["source"] = source_name
    out["sgRNA"] = df[mapping.guide_col].map(clean_sequence)
    out["target"] = df[mapping.target_col].map(clean_sequence)

    if mapping.label_col is not None:
        labels = [_label_from_value(v, positives, negatives) for v in df[mapping.label_col]]
    else:
        labels = [None] * len(df)

    if any(v is None for v in labels) and mapping.score_col is not None:
        numeric_scores = pd.to_numeric(df[mapping.score_col], errors="coerce")
        score_labels = [None if pd.isna(v) else int(float(v) > score_threshold) for v in numeric_scores]
        labels = [score_labels[i] if labels[i] is None else labels[i] for i in range(len(labels))]

    if all(v is None for v in labels):
        raise ValueError(
            "Could not infer labels. Provide --label-col with binary labels, or --score-col with --score-threshold."
        )

    out["label"] = labels
    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)
    out = out[(out["sgRNA"].str.len() >= min_length) & (out["target"].str.len() >= min_length)].copy()
    out = out.drop_duplicates(subset=["sgRNA", "target", "label"]).reset_index(drop=True)
    return out[["sample_id", "source", "sgRNA", "target", "label"]]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def write_standardized(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
