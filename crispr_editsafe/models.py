from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelBundle:
    model: Any
    feature_columns: list[str]
    model_name: str


def build_model(model_name: str = "logistic_regression", random_state: int = 42) -> Any:
    if model_name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ])
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model_name: {model_name}")


def fit_model(X: pd.DataFrame, y: pd.Series, model_name: str = "logistic_regression") -> ModelBundle:
    model = build_model(model_name=model_name)
    model.fit(X, y)
    return ModelBundle(model=model, feature_columns=list(X.columns), model_name=model_name)


def predict_proba(bundle: ModelBundle, X: pd.DataFrame) -> np.ndarray:
    X = X[bundle.feature_columns]
    if hasattr(bundle.model, "predict_proba"):
        return bundle.model.predict_proba(X)[:, 1]
    scores = bundle.model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def save_model(bundle: ModelBundle, path: str) -> None:
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model(path: str) -> ModelBundle:
    return joblib.load(path)
