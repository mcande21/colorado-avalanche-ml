from __future__ import annotations

import numpy as np
import torch
from torch import nn


class DangerLevelGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        out = hidden[-1]
        return self.classifier(out)


def build_s2_sequences(
    features: np.ndarray,
    s1_probs: np.ndarray,
    station_ids: np.ndarray,
    lookback: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    combined = np.hstack([features, s1_probs])
    n_feat = combined.shape[1]

    all_seqs = []
    all_idx = []

    for station in np.unique(station_ids):
        mask = station_ids == station
        station_indices = np.where(mask)[0]
        station_X = combined[station_indices]
        n_rows = len(station_X)
        n_seq = n_rows - lookback
        if n_seq <= 0:
            continue
        idx = np.arange(lookback)[None, :] + np.arange(n_seq)[:, None]
        all_seqs.append(station_X[idx])
        all_idx.append(station_indices[lookback:])

    if not all_seqs:
        return (
            np.zeros((0, lookback, n_feat), dtype=features.dtype),
            np.zeros(0, dtype=np.int64),
        )

    return np.concatenate(all_seqs), np.concatenate(all_idx)
