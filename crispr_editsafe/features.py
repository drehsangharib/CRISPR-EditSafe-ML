from __future__ import annotations

import re
from typing import Iterable
import numpy as np
import pandas as pd

VALID_DNA = re.compile(r"^[ACGTNacgtn]+$")
BASES = ["A", "C", "G", "T"]


def normalize_sequence(seq: str) -> str:
    if not isinstance(seq, str):
        raise TypeError("Sequence must be a string")
    seq = seq.upper().replace("U", "T").strip()
    if not VALID_DNA.match(seq):
        raise ValueError(f"Invalid DNA/RNA sequence: {seq}")
    return seq


def validate_pair(sgrna: str, target: str, min_len: int = 20) -> tuple[str, str]:
    sg = normalize_sequence(sgrna)
    tg = normalize_sequence(target)
    if len(sg) < min_len or len(tg) < min_len:
        raise ValueError("sgRNA and target must be at least 20 nt long")
    return sg[:20], tg[:20]


def gc_content(seq: str) -> float:
    seq = normalize_sequence(seq)
    if len(seq) == 0:
        return 0.0
    return float(sum(base in {"G", "C"} for base in seq) / len(seq))


def mismatch_vector(sgrna: str, target: str) -> list[int]:
    sg, tg = validate_pair(sgrna, target)
    return [int(a != b and a != "N" and b != "N") for a, b in zip(sg, tg)]


def one_hot_sequence(seq: str, prefix: str, length: int = 20) -> dict[str, int]:
    seq = normalize_sequence(seq)[:length]
    values = {}
    for pos in range(length):
        base = seq[pos] if pos < len(seq) else "N"
        for b in BASES:
            values[f"{prefix}_pos{pos+1}_{b}"] = int(base == b)
    return values


def featurize_pair(sgrna: str, target: str) -> dict[str, float | int]:
    sg, tg = validate_pair(sgrna, target)
    mm = mismatch_vector(sg, tg)
    features: dict[str, float | int] = {
        "guide_gc": gc_content(sg),
        "target_gc": gc_content(tg),
        "mismatch_count": int(sum(mm)),
        "seed_mismatch_count_pos1_10": int(sum(mm[:10])),
        "pam_proximal_mismatch_count_pos11_20": int(sum(mm[10:20])),
        "distal_mismatch_count_pos1_5": int(sum(mm[:5])),
        "middle_mismatch_count_pos6_15": int(sum(mm[5:15])),
        "proximal_mismatch_count_pos16_20": int(sum(mm[15:20])),
    }
    for i, val in enumerate(mm, start=1):
        features[f"mismatch_pos{i}"] = int(val)
    features.update(one_hot_sequence(sg, "sgRNA"))
    features.update(one_hot_sequence(tg, "target"))
    return features


def featurize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = [featurize_pair(row.sgRNA, row.target) for row in df.itertuples(index=False)]
    feature_df = pd.DataFrame(rows)
    metadata_cols = [c for c in ["sample_id", "source", "sgRNA", "target", "label"] if c in df.columns]
    return pd.concat([df[metadata_cols].reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"sample_id", "source", "sgRNA", "target", "label", "activity"}
    return [c for c in df.columns if c not in excluded]
