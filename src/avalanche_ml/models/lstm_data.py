"""Hierarchical multi-rate data loader for avalanche LSTM."""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field

import duckdb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from avalanche_ml.features.alignment import (
    ELEVATION_BANDS,
    PROBLEM_TYPE_FLAGS,
    TRAIN_END,
    VAL_END,
)
from avalanche_ml.features.physics import PHYSICS_FEATURES
from avalanche_ml.features.weather import get_feature_columns

MAX_SEASON_STEPS = 60
BRANCH1_DAYS = 7
BRANCH2_DAYS = 30
AGGREGATION_DAYS = 3
MIN_HISTORY_DAYS = 7


def get_sequence_feature_columns() -> list[str]:
    return get_feature_columns() + PHYSICS_FEATURES


def _season_start(d: datetime.date) -> datetime.date:
    if d.month >= 10:
        return datetime.date(d.year, 10, 1)
    return datetime.date(d.year - 1, 10, 1)


@dataclass
class NormalizationStats:
    means: np.ndarray
    stds: np.ndarray
    feature_columns: list[str] = field(default_factory=list)

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        stds = self.stds.copy()
        stds[stds == 0] = 1.0
        return (arr - self.means) / stds


def compute_normalization_stats(
    conn: duckdb.DuckDBPyConnection,
    end_date: datetime.date,
) -> NormalizationStats:
    feature_cols = get_sequence_feature_columns()
    agg_parts = []
    for col in feature_cols:
        safe = f"CASE WHEN isnan({col}) THEN NULL ELSE {col} END"
        agg_parts.append(f"AVG({safe})")
        agg_parts.append(f"STDDEV_POP({safe})")
    agg_sql = ", ".join(agg_parts)
    row = conn.execute(
        f"SELECT {agg_sql} FROM training_matrix WHERE date <= ?", [end_date],
    ).fetchone()

    means = np.zeros(len(feature_cols), dtype=np.float32)
    stds = np.ones(len(feature_cols), dtype=np.float32)
    for i in range(len(feature_cols)):
        m = row[i * 2]
        s = row[i * 2 + 1]
        if m is not None:
            means[i] = float(m)
        if s is not None and float(s) > 0:
            stds[i] = float(s)
    return NormalizationStats(means=means, stds=stds, feature_columns=feature_cols)


class AvalancheSequenceDataset(Dataset):
    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        start_date: datetime.date,
        end_date: datetime.date,
        norm_stats: NormalizationStats | None = None,
        stations: list[str] | None = None,
    ):
        self.conn = conn
        self.norm_stats = norm_stats
        self.feature_cols = get_sequence_feature_columns()
        self.n_features = len(self.feature_cols)

        where = "date >= ? AND date <= ?"
        params: list = [start_date, end_date]
        if stations:
            placeholders = ", ".join("?" for _ in stations)
            where += f" AND station_id IN ({placeholders})"
            params.extend(stations)

        all_triples = conn.execute(f"""
            SELECT DISTINCT station_id, date, elevation_band
            FROM training_matrix
            WHERE {where}
            ORDER BY station_id, date, elevation_band
        """, params).fetchall()

        seen: dict[tuple, int] = {}
        for sid, dt, _ in all_triples:
            key = (sid, dt)
            if key not in seen:
                lb = dt - datetime.timedelta(days=BRANCH1_DAYS - 1)
                cnt = conn.execute("""
                    SELECT COUNT(DISTINCT date) FROM training_matrix
                    WHERE station_id = ? AND date >= ? AND date <= ?
                """, [sid, lb, dt]).fetchone()[0]
                seen[key] = cnt

        self.samples = [
            (sid, dt, eb) for sid, dt, eb in all_triples
            if seen.get((sid, dt), 0) >= MIN_HISTORY_DAYS
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def _query_features(
        self, station_id: str, start_date: datetime.date, end_date: datetime.date,
    ) -> np.ndarray:
        cols_sql = ", ".join(self.feature_cols)
        df = self.conn.execute(f"""
            SELECT {cols_sql}
            FROM training_matrix
            WHERE station_id = ? AND date >= ? AND date <= ?
              AND elevation_band = ?
            ORDER BY date
        """, [station_id, start_date, end_date, ELEVATION_BANDS[0]]).fetchdf()
        if df.empty:
            return np.zeros((0, self.n_features), dtype=np.float32)
        return df.values.astype(np.float32)

    def _build_branch(self, features: np.ndarray, target_len: int) -> torch.Tensor:
        n = len(features)
        if n >= target_len:
            arr = features[-target_len:]
        elif n > 0:
            pad = np.zeros((target_len - n, self.n_features), dtype=np.float32)
            arr = np.vstack([pad, features])
        else:
            arr = np.zeros((target_len, self.n_features), dtype=np.float32)
        if self.norm_stats is not None:
            arr = self.norm_stats.normalize(arr)
        arr = np.nan_to_num(arr, nan=0.0)
        return torch.from_numpy(arr.copy()).float()

    def _build_branch3(
        self, station_id: str, date: datetime.date,
    ) -> tuple[torch.Tensor, int]:
        season_s = _season_start(date)
        features = self._query_features(station_id, season_s, date)
        if len(features) == 0:
            return torch.zeros(MAX_SEASON_STEPS, self.n_features), 0

        n_days = len(features)
        n_groups = (n_days + AGGREGATION_DAYS - 1) // AGGREGATION_DAYS
        agg = np.zeros((n_groups, self.n_features), dtype=np.float32)
        for i in range(n_groups):
            s = i * AGGREGATION_DAYS
            e = min(s + AGGREGATION_DAYS, n_days)
            agg[i] = np.nanmean(features[s:e], axis=0)

        actual_len = n_groups
        if self.norm_stats is not None:
            agg = self.norm_stats.normalize(agg)
        agg = np.nan_to_num(agg, nan=0.0)

        if actual_len < MAX_SEASON_STEPS:
            pad = np.zeros(
                (MAX_SEASON_STEPS - actual_len, self.n_features), dtype=np.float32,
            )
            result = np.vstack([agg, pad])
        else:
            result = agg[:MAX_SEASON_STEPS]
            actual_len = MAX_SEASON_STEPS
        return torch.from_numpy(result.copy()).float(), actual_len

    def __getitem__(self, idx: int) -> dict:
        station_id, date, elevation_band = self.samples[idx]

        b1_s = date - datetime.timedelta(days=BRANCH1_DAYS - 1)
        branch1 = self._build_branch(
            self._query_features(station_id, b1_s, date), BRANCH1_DAYS,
        )

        b2_s = date - datetime.timedelta(days=BRANCH2_DAYS - 1)
        branch2 = self._build_branch(
            self._query_features(station_id, b2_s, date), BRANCH2_DAYS,
        )

        branch3, branch3_len = self._build_branch3(station_id, date)

        problem_cols = ", ".join(PROBLEM_TYPE_FLAGS)
        label_row = self.conn.execute(f"""
            SELECT danger_level, {problem_cols} FROM training_matrix
            WHERE station_id = ? AND date = ? AND elevation_band = ?
            LIMIT 1
        """, [station_id, date, elevation_band]).fetchone()

        danger = label_row[0] if label_row else 0
        flags = list(label_row[1:]) if label_row else [0] * len(PROBLEM_TYPE_FLAGS)

        labels: dict = {
            "danger_level": danger,
            "danger_binary": 1 if danger >= 3 else 0,
        }
        for i, name in enumerate(PROBLEM_TYPE_FLAGS):
            labels[name] = flags[i]

        return {
            "branch1": branch1,
            "branch2": branch2,
            "branch3": branch3,
            "branch3_lengths": branch3_len,
            "labels": labels,
            "metadata": {
                "station_id": station_id,
                "date": str(date),
                "elevation_band": elevation_band,
            },
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "branch1": torch.stack([s["branch1"] for s in batch]),
        "branch2": torch.stack([s["branch2"] for s in batch]),
        "branch3": torch.stack([s["branch3"] for s in batch]),
        "branch3_lengths": torch.tensor(
            [s["branch3_lengths"] for s in batch], dtype=torch.long,
        ),
        "labels": {
            k: torch.tensor([s["labels"][k] for s in batch], dtype=torch.long)
            for k in batch[0]["labels"]
        },
        "metadata": {
            "station_id": [s["metadata"]["station_id"] for s in batch],
            "date": [s["metadata"]["date"] for s in batch],
            "elevation_band": [s["metadata"]["elevation_band"] for s in batch],
        },
    }


def create_dataloaders(
    db_path: str, batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    conn = duckdb.connect(db_path, read_only=True)
    norm_stats = compute_normalization_stats(conn, TRAIN_END)

    train_ds = AvalancheSequenceDataset(
        conn, datetime.date(2000, 1, 1), TRAIN_END, norm_stats=norm_stats,
    )
    val_ds = AvalancheSequenceDataset(
        conn,
        TRAIN_END + datetime.timedelta(days=1),
        VAL_END,
        norm_stats=norm_stats,
    )
    test_ds = AvalancheSequenceDataset(
        conn,
        VAL_END + datetime.timedelta(days=1),
        datetime.date(2099, 12, 31),
        norm_stats=norm_stats,
    )

    return (
        DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        ),
        DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        ),
        DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        ),
    )
