"""Tests for hierarchical LSTM model architecture and training."""
from __future__ import annotations

import json
import math
import os
import tempfile

import torch

N_FEATURES = 98
N_PROBLEM_TYPES = 3
N_DANGER_LEVELS = 5
BATCH_SIZE = 8
HIDDEN_SIZE = 64


def _make_batch(
    batch_size: int = BATCH_SIZE,
    n_features: int = N_FEATURES,
    branch3_lengths: list[int] | None = None,
) -> dict:
    if branch3_lengths is None:
        branch3_lengths = [20] * batch_size
    return {
        "branch1": torch.randn(batch_size, 7, n_features),
        "branch2": torch.randn(batch_size, 30, n_features),
        "branch3": torch.randn(batch_size, 60, n_features),
        "branch3_lengths": torch.tensor(branch3_lengths, dtype=torch.long),
    }


def _make_labels(batch_size: int = BATCH_SIZE) -> dict:
    return {
        "danger_level": torch.randint(1, 6, (batch_size,)),
        "danger_binary": torch.randint(0, 2, (batch_size,)),
        "persistent_slab": torch.randint(0, 2, (batch_size,)),
        "storm_slab": torch.randint(0, 2, (batch_size,)),
        "loose_wet": torch.randint(0, 2, (batch_size,)),
    }


class TestModelInstantiation:
    def test_creates_three_branch_lstms(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        assert hasattr(model, "branch1_lstm")
        assert hasattr(model, "branch2_lstm")
        assert hasattr(model, "branch3_lstm")

    def test_lstm_config(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        for name in ["branch1_lstm", "branch2_lstm", "branch3_lstm"]:
            lstm = getattr(model, name)
            assert lstm.input_size == N_FEATURES
            assert lstm.hidden_size == HIDDEN_SIZE
            assert lstm.num_layers == 2
            assert lstm.batch_first is True

    def test_trunk_layers(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        assert hasattr(model, "trunk")

    def test_stage1_head(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        assert hasattr(model, "stage1_head")

    def test_stage2_binary_head(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        assert hasattr(model, "stage2_binary_head")

    def test_stage2_multi_head(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        assert hasattr(model, "stage2_multi_head")

    def test_default_parameters(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0


class TestForwardPass:
    def test_output_keys(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            out = model(**batch)
        assert "problem_types" in out
        assert "danger_binary" in out
        assert "danger_distribution" in out

    def test_output_shapes(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            out = model(**batch)
        assert out["problem_types"].shape == (BATCH_SIZE, N_PROBLEM_TYPES)
        assert out["danger_binary"].shape == (BATCH_SIZE, 1)
        assert out["danger_distribution"].shape == (BATCH_SIZE, N_DANGER_LEVELS)

    def test_problem_types_range(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            out = model(**batch)
        assert (out["problem_types"] >= 0).all()
        assert (out["problem_types"] <= 1).all()

    def test_danger_binary_range(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            out = model(**batch)
        assert (out["danger_binary"] >= 0).all()
        assert (out["danger_binary"] <= 1).all()

    def test_danger_distribution_sums_to_one(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        batch = _make_batch()
        with torch.no_grad():
            out = model(**batch)
        sums = out["danger_distribution"].sum(dim=1)
        assert torch.allclose(sums, torch.ones(BATCH_SIZE), atol=1e-5)

    def test_variable_branch3_lengths(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        lengths = [5, 10, 20, 30, 40, 50, 60, 15]
        batch = _make_batch(branch3_lengths=lengths)
        with torch.no_grad():
            out = model(**batch)
        assert out["problem_types"].shape == (BATCH_SIZE, N_PROBLEM_TYPES)

    def test_stage2_receives_stage1_output(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        batch = _make_batch(batch_size=4)
        out = model(**batch)

        loss = out["problem_types"].sum() + out["danger_binary"].sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.stage1_head.parameters()
        )
        assert has_grad, "Stage 1 head should receive gradients from Stage 2 path"

    def test_zero_length_branch3(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM

        model = HierarchicalLSTM()
        model.eval()
        lengths = [0, 10, 5, 20, 15, 30, 25, 8]
        batch = _make_batch(branch3_lengths=lengths)
        with torch.no_grad():
            out = model(**batch)
        assert not torch.isnan(out["problem_types"]).any()
        assert not torch.isnan(out["danger_binary"]).any()


class TestFocalLoss:
    def test_instantiation(self):
        from avalanche_ml.models.lstm_train import FocalLoss

        fl = FocalLoss(gamma=2.0)
        assert fl.gamma == 2.0

    def test_known_output(self):
        from avalanche_ml.models.lstm_train import FocalLoss

        fl = FocalLoss(gamma=0.0)
        pred = torch.tensor([0.8, 0.2], dtype=torch.float32)
        target = torch.tensor([1.0, 0.0], dtype=torch.float32)
        loss = fl(pred, target)
        expected = -(math.log(0.8) + math.log(0.8)) / 2
        assert abs(loss.item() - expected) < 1e-4

    def test_gamma_reduces_easy_examples(self):
        from avalanche_ml.models.lstm_train import FocalLoss

        fl_0 = FocalLoss(gamma=0.0)
        fl_2 = FocalLoss(gamma=2.0)
        pred = torch.tensor([0.9], dtype=torch.float32)
        target = torch.tensor([1.0], dtype=torch.float32)
        loss_0 = fl_0(pred, target)
        loss_2 = fl_2(pred, target)
        assert loss_2 < loss_0

    def test_with_alpha_weights(self):
        from avalanche_ml.models.lstm_train import FocalLoss

        alpha = torch.tensor([0.7, 0.3])
        fl = FocalLoss(gamma=2.0, alpha=alpha)
        pred = torch.tensor([0.5, 0.5], dtype=torch.float32)
        target = torch.tensor([1.0, 1.0], dtype=torch.float32)
        loss = fl(pred, target)
        assert loss.item() > 0


class TestCombinedLoss:
    def test_weighted_sum(self):
        from avalanche_ml.models.lstm_train import combined_loss

        out = {
            "problem_types": torch.tensor([[0.5, 0.5, 0.5]]),
            "danger_binary": torch.tensor([[0.5]]),
            "danger_distribution": torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]]),
        }
        labels = {
            "persistent_slab": torch.tensor([1]),
            "storm_slab": torch.tensor([0]),
            "loose_wet": torch.tensor([1]),
            "danger_binary": torch.tensor([1]),
            "danger_level": torch.tensor([3]),
        }
        loss = combined_loss(out, labels)
        assert loss.item() > 0


class TestTrainingLoop:
    def _make_loader(self, n_samples: int = 50, batch_size: int = 10):
        from torch.utils.data import DataLoader

        b1 = torch.randn(n_samples, 7, N_FEATURES)
        b2 = torch.randn(n_samples, 30, N_FEATURES)
        b3 = torch.randn(n_samples, 60, N_FEATURES)
        b3_len = torch.randint(5, 60, (n_samples,))
        danger = torch.randint(1, 6, (n_samples,))
        danger_bin = (danger >= 3).long()
        ps = torch.randint(0, 2, (n_samples,))
        ss = torch.randint(0, 2, (n_samples,))
        lw = torch.randint(0, 2, (n_samples,))

        class _DS:
            def __init__(self):
                self.length = n_samples

            def __len__(self):
                return self.length

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
                        "elevation_band": "above_treeline",
                    },
                }

        from avalanche_ml.models.lstm_data import collate_fn

        return DataLoader(_DS(), batch_size=batch_size, collate_fn=collate_fn)

    def test_train_runs_n_epochs(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM
        from avalanche_ml.models.lstm_train import train_lstm

        model = HierarchicalLSTM()
        train_loader = self._make_loader()
        val_loader = self._make_loader(n_samples=20)
        _, history = train_lstm(
            model, train_loader, val_loader, epochs=3, device="cpu",
        )
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3

    def test_validation_loss_tracked(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM
        from avalanche_ml.models.lstm_train import train_lstm

        model = HierarchicalLSTM()
        train_loader = self._make_loader()
        val_loader = self._make_loader(n_samples=20)
        _, history = train_lstm(
            model, train_loader, val_loader, epochs=3, device="cpu",
        )
        for v in history["val_loss"]:
            assert v > 0

    def test_early_stopping(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM
        from avalanche_ml.models.lstm_train import train_lstm

        model = HierarchicalLSTM()
        train_loader = self._make_loader(n_samples=10, batch_size=10)
        val_loader = self._make_loader(n_samples=10, batch_size=10)
        _, history = train_lstm(
            model,
            train_loader,
            val_loader,
            epochs=100,
            patience=2,
            device="cpu",
        )
        assert len(history["train_loss"]) <= 100

    def test_lr_scheduler_reduces_lr(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM
        from avalanche_ml.models.lstm_train import train_lstm

        model = HierarchicalLSTM()
        train_loader = self._make_loader()
        val_loader = self._make_loader(n_samples=20)
        _, history = train_lstm(
            model,
            train_loader,
            val_loader,
            epochs=3,
            device="cpu",
        )
        assert "lr" in history

    def test_gradient_clipping(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM
        from avalanche_ml.models.lstm_train import train_lstm

        model = HierarchicalLSTM()
        train_loader = self._make_loader()
        val_loader = self._make_loader(n_samples=20)
        _, history = train_lstm(
            model,
            train_loader,
            val_loader,
            epochs=1,
            max_grad_norm=1.0,
            device="cpu",
        )
        assert "max_grad_norm" in history


class TestModelSerialization:
    def test_save_load_roundtrip(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM
        from avalanche_ml.models.lstm_train import load_model, save_model

        model = HierarchicalLSTM()
        model.eval()
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            out_before = model(**batch)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            save_model(model, path, {"epochs": 10, "val_loss": 0.5})
            loaded = load_model(path)

        loaded.eval()
        with torch.no_grad():
            out_after = loaded(**batch)

        assert torch.allclose(
            out_before["problem_types"], out_after["problem_types"], atol=1e-6,
        )
        assert torch.allclose(
            out_before["danger_binary"], out_after["danger_binary"], atol=1e-6,
        )

    def test_metadata_saved(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM
        from avalanche_ml.models.lstm_train import save_model

        model = HierarchicalLSTM()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            save_model(model, path, {"epochs": 10})
            meta_path = path.replace(".pt", "_metadata.json")
            assert os.path.exists(meta_path)
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["epochs"] == 10


class TestPrediction:
    def test_predict_returns_dataframe(self):
        from avalanche_ml.models.lstm_model import HierarchicalLSTM
        from avalanche_ml.models.lstm_train import predict

        model = HierarchicalLSTM()
        model.eval()

        class _Loader:
            def __iter__(self):
                batch = _make_batch(batch_size=4)
                batch["labels"] = _make_labels(batch_size=4)
                batch["metadata"] = {
                    "station_id": ["S1"] * 4,
                    "date": ["2023-01-01"] * 4,
                    "elevation_band": ["above_treeline"] * 4,
                }
                yield batch

        df = predict(model, _Loader(), device="cpu")
        assert len(df) == 4
        assert "prob_persistent_slab" in df.columns
        assert "prob_storm_slab" in df.columns
        assert "prob_loose_wet" in df.columns
        assert "prob_danger_binary" in df.columns
        for i in range(N_DANGER_LEVELS):
            assert f"prob_danger_{i + 1}" in df.columns
