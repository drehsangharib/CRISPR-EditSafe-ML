from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_TRAIN_COLUMNS = {"sgRNA", "target", "label"}
REQUIRED_INFERENCE_COLUMNS = {"sgRNA", "target"}


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def load_training_data(path: str | Path) -> pd.DataFrame:
    df = read_csv(path)
    ensure_columns(df, REQUIRED_TRAIN_COLUMNS)
    return df


def load_inference_data(path: str | Path) -> pd.DataFrame:
    df = read_csv(path)
    ensure_columns(df, REQUIRED_INFERENCE_COLUMNS)
    return df


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
