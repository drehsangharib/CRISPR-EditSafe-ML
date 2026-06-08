from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
import pandas as pd

BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def _clean_seq(seq: object, length: int = 20) -> str:
    s = str(seq).upper().replace("U", "T").replace("_", "").replace("-", "").replace(".", "")
    s = "".join(ch if ch in BASE_TO_INDEX else "N" for ch in s)
    if len(s) < length:
        s = s + "N" * (length - len(s))
    return s[:length]


def encode_sequence_pair(sgrna: object, target: object, length: int = 20) -> np.ndarray:
    """Encode an sgRNA/target pair as channels x length for CNN input.

    Channels:
      0-4: one-hot guide bases A,C,G,T,N
      5-9: one-hot target bases A,C,G,T,N
      10: mismatch indicator
    """
    guide = _clean_seq(sgrna, length=length)
    target = _clean_seq(target, length=length)
    arr = np.zeros((11, length), dtype=np.float32)
    for i, (g, t) in enumerate(zip(guide, target)):
        gi = BASE_TO_INDEX.get(g, 4)
        ti = BASE_TO_INDEX.get(t, 4)
        arr[gi, i] = 1.0
        arr[5 + ti, i] = 1.0
        arr[10, i] = 1.0 if gi != ti and gi != 4 and ti != 4 else 0.0
    return arr


def encode_dataframe_pairs(df: pd.DataFrame, length: int = 20) -> np.ndarray:
    if "sgRNA" not in df.columns or "target" not in df.columns:
        raise ValueError("Input DataFrame must contain sgRNA and target columns.")
    encoded = [encode_sequence_pair(row.sgRNA, row.target, length=length) for row in df.itertuples(index=False)]
    return np.stack(encoded, axis=0)


@dataclass
class CNNTrainingResult:
    history: list[dict]
    best_validation_loss: float


def require_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the CNN model. Install CPU PyTorch with: pip install torch"
        ) from exc


class CRISPROffTargetCNN:  # actual superclass assigned dynamically in __new__ not possible; use factory below
    pass


def build_cnn_model(sequence_length: int = 20, n_channels: int = 11, hidden_dim: int = 64, dropout: float = 0.25):
    torch, nn = require_torch()

    class _CRISPROffTargetCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.AdaptiveMaxPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(64, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x):
            return self.network(x).squeeze(-1)

    return _CRISPROffTargetCNN()


def predict_cnn_proba(model, X, batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    torch, _nn = require_torch()
    model.eval()
    model.to(device)
    probs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_cnn(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    random_state: int = 42,
    device: str = "cpu",
):
    torch, nn = require_torch()
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    model = build_cnn_model(sequence_length=X_train.shape[-1], n_channels=X_train.shape[1])
    model.to(device)

    y_train_np = np.asarray(y_train).astype(np.float32)
    positives = float(y_train_np.sum())
    negatives = float(len(y_train_np) - positives)
    pos_weight_value = negatives / max(positives, 1.0)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(np.asarray(y_val).astype(np.float32), dtype=torch.float32, device=device)

    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(X_train_t.shape[0])
        train_losses = []
        for start in range(0, X_train_t.shape[0], batch_size):
            idx = permutation[start:start + batch_size]
            xb = X_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(criterion(val_logits, y_val_t).detach().cpu())

        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "validation_loss": val_loss,
        }
        history.append(record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, CNNTrainingResult(history=history, best_validation_loss=best_val_loss)
