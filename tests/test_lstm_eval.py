"""Tests for LSTM evaluation pipeline."""
from __future__ import annotations

import torch

from avalanche_ml.features.alignment import PROBLEM_TYPE_FLAGS

N_FEATURES = 98
N_PROBLEM_TYPES = 3
N_DANGER_LEVELS = 5


def _make_batch(batch_size: int = 8) -> dict:
    return {
        "branch1": torch.randn(batch_size, 7, N_FEATURES),
        "branch2": torch.randn(batch_size, 30, N_FEATURES),
        "branch3": torch.randn(batch_size, 60, N_FEATURES),
        "branch3_lengths": torch.tensor([20] * batch_size, dtype=torch.long),
    }


def _make_labels(batch_size: int = 8) -> dict:
    return {
        "danger_level": torch.randint(1, 6, (batch_size,)),
        "danger_binary": torch.randint(0, 2, (batch_size,)),
        "persistent_slab": torch.randint(0, 2, (batch_size,)),
        "storm_slab": torch.randint(0, 2, (batch_size,)),
        "loose_wet": torch.randint(0, 2, (batch_size,)),
    }


class _FakeLoader:
    def __init__(self, n_batches: int = 3, batch_size: int = 8):
        self.n_batches = n_batches
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = _make_batch(self.batch_size)
            batch["labels"] = _make_labels(self.batch_size)
            batch["metadata"] = {
                "station_id": ["S1"] * self.batch_size,
                "date": ["2023-01-01"] * self.batch_size,
                "elevation_band": ["above_treeline"] * (self.batch_size // 2)
                    + ["below_treeline"] * (self.batch_size - self.batch_size // 2),
            }
            yield batch


class TestEvaluateLSTM:
    def test_returns_stage1_and_stage2(self):
        from avalanche_ml.models.lstm_eval import evaluate_lstm
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        loader = _FakeLoader(n_batches=2, batch_size=8)
        result = evaluate_lstm(model, loader, device="cpu")
        assert "stage1" in result
        assert "stage2" in result

    def test_stage1_per_type_metrics(self):
        from avalanche_ml.models.lstm_eval import evaluate_lstm
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        loader = _FakeLoader(n_batches=2, batch_size=8)
        result = evaluate_lstm(model, loader, device="cpu")
        for pt in PROBLEM_TYPE_FLAGS:
            assert pt in result["stage1"]["per_type"]
            assert "precision" in result["stage1"]["per_type"][pt]
            assert "recall" in result["stage1"]["per_type"][pt]
            assert "f1" in result["stage1"]["per_type"][pt]

    def test_stage2_binary_f1(self):
        from avalanche_ml.models.lstm_eval import evaluate_lstm
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        loader = _FakeLoader(n_batches=2, batch_size=8)
        result = evaluate_lstm(model, loader, device="cpu")
        assert "binary_f1" in result["stage2"]
        assert 0.0 <= result["stage2"]["binary_f1"] <= 1.0

    def test_stage2_ordinal_accuracy(self):
        from avalanche_ml.models.lstm_eval import evaluate_lstm
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        loader = _FakeLoader(n_batches=2, batch_size=8)
        result = evaluate_lstm(model, loader, device="cpu")
        assert "ordinal_accuracy" in result["stage2"]

    def test_stage2_confusion_matrix(self):
        from avalanche_ml.models.lstm_eval import evaluate_lstm
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        loader = _FakeLoader(n_batches=2, batch_size=8)
        result = evaluate_lstm(model, loader, device="cpu")
        assert "confusion_matrix" in result["stage2"]
        cm = result["stage2"]["confusion_matrix"]
        assert cm.shape == (5, 5)

    def test_stage2_false_negative_rate(self):
        from avalanche_ml.models.lstm_eval import evaluate_lstm
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        loader = _FakeLoader(n_batches=2, batch_size=8)
        result = evaluate_lstm(model, loader, device="cpu")
        assert "false_negative_rate" in result["stage2"]
        assert "high_danger_detection_rate" in result["stage2"]

    def test_stage2_brier_score(self):
        from avalanche_ml.models.lstm_eval import evaluate_lstm
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        loader = _FakeLoader(n_batches=2, batch_size=8)
        result = evaluate_lstm(model, loader, device="cpu")
        assert "brier_score" in result["stage2"]
        assert 0.0 <= result["stage2"]["brier_score"] <= 1.0

    def test_per_elevation_band(self):
        from avalanche_ml.models.lstm_eval import evaluate_lstm
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        loader = _FakeLoader(n_batches=2, batch_size=8)
        result = evaluate_lstm(model, loader, device="cpu")
        assert "per_elevation_band" in result
        assert len(result["per_elevation_band"]) > 0

    def test_metric_keys_match_rf_format(self):
        from avalanche_ml.models.lstm_eval import evaluate_lstm
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        loader = _FakeLoader(n_batches=2, batch_size=8)
        result = evaluate_lstm(model, loader, device="cpu")
        rf_stage2_keys = {
            "binary_f1", "binary_precision", "binary_recall",
            "ordinal_accuracy", "confusion_matrix", "macro_f1",
            "brier_score", "false_negative_rate", "high_danger_detection_rate",
        }
        assert rf_stage2_keys == set(result["stage2"].keys())


class TestRunLSTMTrainingPipeline:
    def _make_loader(self, n_samples: int = 30, batch_size: int = 10):
        from torch.utils.data import DataLoader

        from avalanche_ml.models.lstm_data import collate_fn

        b1 = torch.randn(n_samples, 7, N_FEATURES)
        b2 = torch.randn(n_samples, 30, N_FEATURES)
        b3 = torch.randn(n_samples, 60, N_FEATURES)
        b3_len = torch.randint(5, 60, (n_samples,))
        danger = torch.randint(1, 6, (n_samples,))
        danger_bin = (danger >= 3).long()
        ps = torch.randint(0, 2, (n_samples,))
        ss = torch.randint(0, 2, (n_samples,))
        lw = torch.randint(0, 2, (n_samples,))
        bands = ["above_treeline"] * (n_samples // 2) \
            + ["below_treeline"] * (n_samples - n_samples // 2)

        class _DS:
            def __len__(self):
                return n_samples

            def __getitem__(self, idx):
                return {
                    "branch1": b1[idx],
                    "branch2": b2[idx],
                    "branch3": b3[idx],
                    "branch3_lengths": b3_len[idx],
                    "labels": {
                        "danger_level": danger[idx],
                        "danger_binary": danger_bin[idx],
                        "persistent_slab": ps[idx],
                        "storm_slab": ss[idx],
                        "loose_wet": lw[idx],
                    },
                    "metadata": {
                        "station_id": "S1",
                        "date": "2023-01-01",
                        "elevation_band": bands[idx],
                    },
                }

        return DataLoader(_DS(), batch_size=batch_size, collate_fn=collate_fn)

    def test_pipeline_returns_all_splits(self, tmp_path):
        from avalanche_ml.models.lstm_eval import run_lstm_training_pipeline

        train_loader = self._make_loader()
        val_loader = self._make_loader(n_samples=20)
        test_loader = self._make_loader(n_samples=20)
        result = run_lstm_training_pipeline(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=3,
            mlflow_tracking_dir=str(tmp_path / "mlruns"),
            device="cpu",
        )
        assert "train_metrics" in result
        assert "val_metrics" in result
        assert "test_metrics" in result

    def test_pipeline_returns_mlflow_run_id(self, tmp_path):
        from avalanche_ml.models.lstm_eval import run_lstm_training_pipeline

        train_loader = self._make_loader()
        val_loader = self._make_loader(n_samples=20)
        test_loader = self._make_loader(n_samples=20)
        result = run_lstm_training_pipeline(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=3,
            mlflow_tracking_dir=str(tmp_path / "mlruns"),
            device="cpu",
        )
        assert "mlflow_run_id" in result
        assert result["mlflow_run_id"] is not None

    def test_pipeline_returns_history(self, tmp_path):
        from avalanche_ml.models.lstm_eval import run_lstm_training_pipeline

        train_loader = self._make_loader()
        val_loader = self._make_loader(n_samples=20)
        test_loader = self._make_loader(n_samples=20)
        result = run_lstm_training_pipeline(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=3,
            mlflow_tracking_dir=str(tmp_path / "mlruns"),
            device="cpu",
        )
        assert "history" in result
        assert len(result["history"]["train_loss"]) > 0

    def test_pipeline_metrics_match_rf_format(self, tmp_path):
        from avalanche_ml.models.lstm_eval import run_lstm_training_pipeline

        train_loader = self._make_loader()
        val_loader = self._make_loader(n_samples=20)
        test_loader = self._make_loader(n_samples=20)
        result = run_lstm_training_pipeline(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=3,
            mlflow_tracking_dir=str(tmp_path / "mlruns"),
            device="cpu",
        )
        for split in ["train_metrics", "val_metrics", "test_metrics"]:
            m = result[split]
            assert "stage1" in m
            assert "stage2" in m
            assert "binary_f1" in m["stage2"]
