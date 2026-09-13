"""Tests for hierarchical multi-rate LSTM data loader."""
from __future__ import annotations

import datetime
import math

import duckdb
import numpy as np
import pytest
import torch

from avalanche_ml.features.alignment import (
    ELEVATION_BANDS,
    PROBLEM_TYPE_FLAGS,
    create_training_matrix_table,
)
from avalanche_ml.features.physics import PHYSICS_FEATURES
from avalanche_ml.features.weather import get_feature_columns

N_WEATHER = len(get_feature_columns())
N_PHYSICS = len(PHYSICS_FEATURES)
N_FEATURES = N_WEATHER + N_PHYSICS


def _season_start(d: datetime.date) -> datetime.date:
    if d.month >= 10:
        return datetime.date(d.year, 10, 1)
    return datetime.date(d.year - 1, 10, 1)


def _insert_data(
    conn,
    station_id: str,
    start_date: datetime.date,
    n_days: int,
    *,
    elevation_bands=ELEVATION_BANDS,
    zone_id: str = "ZONE_01",
    seed: int = 42,
    feature_value_fn=None,
):
    rng = np.random.RandomState(seed)
    weather_cols = get_feature_columns()
    physics_cols = PHYSICS_FEATURES
    all_feat = weather_cols + physics_cols

    base = ["station_id", "date", "elevation_band", "zone_id"]
    lab = ["danger_level"] + list(PROBLEM_TYPE_FLAGS)
    meta = [
        "elevation", "latitude", "longitude",
        "month", "day_of_year", "days_since_season_start",
    ]
    all_cols = base + all_feat + lab + meta
    cols_str = ", ".join(all_cols)
    ph = ", ".join(["?"] * len(all_cols))

    for d in range(n_days):
        date = start_date + datetime.timedelta(days=d)
        if feature_value_fn is not None:
            fv = [float(feature_value_fn(d))] * len(all_feat)
        else:
            fv = rng.randn(len(all_feat)).tolist()
        dl = int(rng.randint(1, 6))
        pf = [int(rng.randint(0, 2)) for _ in PROBLEM_TYPE_FLAGS]
        ss = _season_start(date)
        for eb in elevation_bands:
            vals = [station_id, date, eb, zone_id]
            vals.extend(fv)
            vals.append(dl)
            vals.extend(pf)
            vals.extend([
                10000.0, 39.5, -105.5,
                date.month, date.timetuple().tm_yday, (date - ss).days,
            ])
            conn.execute(
                f"INSERT INTO training_matrix ({cols_str}) VALUES ({ph})", vals,
            )


@pytest.fixture
def seq_db():
    conn = duckdb.connect(":memory:")
    create_training_matrix_table(conn)
    _insert_data(conn, "S1", datetime.date(2022, 10, 1), 92)
    yield conn
    conn.close()


class TestFeatureColumns:
    def test_sequence_feature_count(self):
        from avalanche_ml.models.lstm_data import get_sequence_feature_columns

        cols = get_sequence_feature_columns()
        assert len(cols) == N_FEATURES
        assert cols[:N_WEATHER] == get_feature_columns()
        assert cols[N_WEATHER:] == PHYSICS_FEATURES


class TestDatasetLength:
    def test_eligible_sample_count(self, seq_db):
        from avalanche_ml.models.lstm_data import AvalancheSequenceDataset

        ds = AvalancheSequenceDataset(
            seq_db, datetime.date(2022, 11, 1), datetime.date(2022, 11, 30),
        )
        assert len(ds) == 90

    def test_short_history_skipped(self):
        from avalanche_ml.models.lstm_data import AvalancheSequenceDataset

        conn = duckdb.connect(":memory:")
        create_training_matrix_table(conn)
        _insert_data(conn, "SHORT", datetime.date(2022, 10, 1), 5)
        ds = AvalancheSequenceDataset(
            conn, datetime.date(2022, 10, 1), datetime.date(2022, 10, 5),
        )
        assert len(ds) == 0
        conn.close()


class TestBranchShapes:
    def test_branch1_shape(self, seq_db):
        from avalanche_ml.models.lstm_data import AvalancheSequenceDataset

        ds = AvalancheSequenceDataset(
            seq_db, datetime.date(2022, 11, 1), datetime.date(2022, 11, 30),
        )
        s = ds[0]
        assert s["branch1"].shape == (7, N_FEATURES)
        assert s["branch1"].dtype == torch.float32

    def test_branch2_shape(self, seq_db):
        from avalanche_ml.models.lstm_data import AvalancheSequenceDataset

        ds = AvalancheSequenceDataset(
            seq_db, datetime.date(2022, 11, 1), datetime.date(2022, 11, 30),
        )
        assert ds[0]["branch2"].shape == (30, N_FEATURES)

    def test_branch3_shape(self, seq_db):
        from avalanche_ml.models.lstm_data import MAX_SEASON_STEPS, AvalancheSequenceDataset

        ds = AvalancheSequenceDataset(
            seq_db, datetime.date(2022, 11, 1), datetime.date(2022, 11, 30),
        )
        assert ds[0]["branch3"].shape == (MAX_SEASON_STEPS, N_FEATURES)


class TestBranch3:
    def test_branch3_lengths(self, seq_db):
        from avalanche_ml.models.lstm_data import AvalancheSequenceDataset

        ds = AvalancheSequenceDataset(
            seq_db, datetime.date(2022, 11, 15), datetime.date(2022, 11, 15),
        )
        s = ds[0]
        assert s["branch3_lengths"] == math.ceil(46 / 3)

    def test_padding_is_zeros(self, seq_db):
        from avalanche_ml.models.lstm_data import AvalancheSequenceDataset

        ds = AvalancheSequenceDataset(
            seq_db, datetime.date(2022, 11, 15), datetime.date(2022, 11, 15),
        )
        s = ds[0]
        assert torch.all(s["branch3"][s["branch3_lengths"]:] == 0)

    def test_3day_aggregation_values(self):
        from avalanche_ml.models.lstm_data import AvalancheSequenceDataset

        conn = duckdb.connect(":memory:")
        create_training_matrix_table(conn)
        _insert_data(
            conn, "AGG", datetime.date(2022, 10, 1), 9,
            feature_value_fn=lambda d: d + 1,
        )
        ds = AvalancheSequenceDataset(
            conn, datetime.date(2022, 10, 9), datetime.date(2022, 10, 9),
        )
        s = ds[0]
        assert s["branch3_lengths"] == 3
        assert abs(s["branch3"][0, 0].item() - 2.0) < 1e-5
        assert abs(s["branch3"][1, 0].item() - 5.0) < 1e-5
        assert abs(s["branch3"][2, 0].item() - 8.0) < 1e-5
        conn.close()

    def test_season_starts_oct1(self):
        from avalanche_ml.models.lstm_data import AvalancheSequenceDataset

        conn = duckdb.connect(":memory:")
        create_training_matrix_table(conn)
        _insert_data(conn, "SEA", datetime.date(2022, 9, 25), 20)
        ds = AvalancheSequenceDataset(
            conn, datetime.date(2022, 10, 14), datetime.date(2022, 10, 14),
        )
        s = ds[0]
        assert s["branch3_lengths"] == math.ceil(14 / 3)
        conn.close()


class TestNormalization:
    def test_stats_computed(self, seq_db):
        from avalanche_ml.models.lstm_data import compute_normalization_stats

        stats = compute_normalization_stats(seq_db, datetime.date(2022, 10, 31))
        assert len(stats.means) == N_FEATURES
        assert len(stats.stds) == N_FEATURES
        assert np.all(stats.stds > 0)

    def test_normalized_near_zero_mean(self, seq_db):
        from avalanche_ml.models.lstm_data import (
            AvalancheSequenceDataset,
            compute_normalization_stats,
        )

        stats = compute_normalization_stats(seq_db, datetime.date(2022, 10, 31))
        ds = AvalancheSequenceDataset(
            seq_db, datetime.date(2022, 10, 8), datetime.date(2022, 10, 31),
            norm_stats=stats,
        )
        feats = np.concatenate(
            [ds[i]["branch1"].numpy() for i in range(len(ds))], axis=0,
        )
        assert np.all(np.abs(np.nanmean(feats, axis=0)) < 1.5)

    def test_no_data_leakage(self, seq_db):
        from avalanche_ml.models.lstm_data import (
            AvalancheSequenceDataset,
            compute_normalization_stats,
        )

        train_stats = compute_normalization_stats(seq_db, datetime.date(2022, 10, 31))
        all_stats = compute_normalization_stats(seq_db, datetime.date(2022, 12, 31))
        assert not np.allclose(train_stats.means, all_stats.means)
        val_ds = AvalancheSequenceDataset(
            seq_db, datetime.date(2022, 11, 1), datetime.date(2022, 11, 30),
            norm_stats=train_stats,
        )
        assert val_ds.norm_stats is train_stats


class TestMissingData:
    def test_nan_filled(self):
        from avalanche_ml.models.lstm_data import AvalancheSequenceDataset

        conn = duckdb.connect(":memory:")
        create_training_matrix_table(conn)
        _insert_data(conn, "NAN", datetime.date(2022, 10, 1), 10)
        conn.execute("""
            UPDATE training_matrix
            SET temp_mean_24h = NULL, temp_min_24h = NULL
            WHERE station_id = 'NAN' AND date = '2022-10-05'
        """)
        ds = AvalancheSequenceDataset(
            conn, datetime.date(2022, 10, 7), datetime.date(2022, 10, 10),
        )
        s = ds[0]
        assert not torch.isnan(s["branch1"]).any()
        assert not torch.isnan(s["branch3"]).any()
        conn.close()


class TestCollate:
    def test_batch_shapes(self, seq_db):
        from avalanche_ml.models.lstm_data import (
            MAX_SEASON_STEPS,
            AvalancheSequenceDataset,
            collate_fn,
        )

        ds = AvalancheSequenceDataset(
            seq_db, datetime.date(2022, 11, 1), datetime.date(2022, 11, 30),
        )
        batch = collate_fn([ds[i] for i in range(4)])
        assert batch["branch1"].shape == (4, 7, N_FEATURES)
        assert batch["branch2"].shape == (4, 30, N_FEATURES)
        assert batch["branch3"].shape == (4, MAX_SEASON_STEPS, N_FEATURES)
        assert batch["branch3_lengths"].shape == (4,)
        assert batch["labels"]["danger_level"].shape == (4,)
        assert len(batch["metadata"]["station_id"]) == 4


class TestDataLoader:
    def test_iteration(self, seq_db):
        from avalanche_ml.models.lstm_data import AvalancheSequenceDataset, collate_fn

        ds = AvalancheSequenceDataset(
            seq_db, datetime.date(2022, 11, 1), datetime.date(2022, 11, 30),
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=8, shuffle=False, collate_fn=collate_fn,
        )
        batch = next(iter(loader))
        assert batch["branch1"].shape[0] == 8

    def test_create_dataloaders(self, tmp_path):
        from avalanche_ml.models.lstm_data import create_dataloaders

        db_path = str(tmp_path / "test.duckdb")
        conn = duckdb.connect(db_path)
        create_training_matrix_table(conn)
        _insert_data(conn, "DL", datetime.date(2023, 6, 15), 16, seed=100)
        _insert_data(conn, "DL", datetime.date(2023, 7, 1), 15, seed=101)
        _insert_data(conn, "DL", datetime.date(2024, 7, 1), 15, seed=102)
        _insert_data(conn, "DL", datetime.date(2025, 2, 1), 15, seed=103)
        conn.close()
        train_ld, val_ld, test_ld = create_dataloaders(db_path, batch_size=4)
        assert len(train_ld.dataset) > 0
        assert len(val_ld.dataset) > 0
        assert len(test_ld.dataset) > 0


class TestLabels:
    def test_label_keys(self, seq_db):
        from avalanche_ml.models.lstm_data import AvalancheSequenceDataset

        ds = AvalancheSequenceDataset(
            seq_db, datetime.date(2022, 11, 1), datetime.date(2022, 11, 30),
        )
        s = ds[0]
        assert "danger_level" in s["labels"]
        assert "danger_binary" in s["labels"]
        for flag in PROBLEM_TYPE_FLAGS:
            assert flag in s["labels"]

    def test_danger_binary_encoding(self, seq_db):
        from avalanche_ml.models.lstm_data import AvalancheSequenceDataset

        ds = AvalancheSequenceDataset(
            seq_db, datetime.date(2022, 11, 1), datetime.date(2022, 11, 30),
        )
        for i in range(min(20, len(ds))):
            s = ds[i]
            dl = s["labels"]["danger_level"]
            assert s["labels"]["danger_binary"] == (1 if dl >= 3 else 0)
