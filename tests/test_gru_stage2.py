from __future__ import annotations

import numpy as np
import torch

from avalanche_ml.features.alignment import PROBLEM_TYPE_FLAGS
from avalanche_ml.features.physics import PHYSICS_FEATURES
from avalanche_ml.features.weather import get_feature_columns


def _n_base_features() -> int:
    return len(get_feature_columns()) + len(PHYSICS_FEATURES)


class TestDangerLevelGRUOutputShape:
    def test_output_shape_is_batch_by_4(self):
        from avalanche_ml.models.gru_stage2 import DangerLevelGRU

        input_size = _n_base_features() + len(PROBLEM_TYPE_FLAGS)
        model = DangerLevelGRU(input_size=input_size)
        x = torch.randn(16, 7, input_size)
        out = model(x)
        assert out.shape == (16, 4)

    def test_single_sample(self):
        from avalanche_ml.models.gru_stage2 import DangerLevelGRU

        input_size = 20
        model = DangerLevelGRU(input_size=input_size)
        x = torch.randn(1, 7, input_size)
        out = model(x)
        assert out.shape == (1, 4)


class TestDangerLevelGRUPredictionRange:
    def test_softmax_probabilities_sum_to_one(self):
        from avalanche_ml.models.gru_stage2 import DangerLevelGRU

        input_size = _n_base_features() + len(PROBLEM_TYPE_FLAGS)
        model = DangerLevelGRU(input_size=input_size)
        model.eval()
        x = torch.randn(32, 7, input_size)
        with torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        np.testing.assert_allclose(sums.numpy(), 1.0, atol=1e-5)


class TestDangerLevelGRUTrains:
    def test_loss_decreases_over_epochs(self):
        from avalanche_ml.models.gru_stage2 import DangerLevelGRU

        input_size = 20
        model = DangerLevelGRU(input_size=input_size, hidden_size=32, num_layers=1)
        x = torch.randn(64, 7, input_size)
        y = torch.randint(0, 4, (64,))

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []
        for _ in range(3):
            model.train()
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses}"
        )


class TestGRUReceivesS1ProbsInSequence:
    def test_input_includes_s1_probs(self):
        from avalanche_ml.models.gru_stage2 import DangerLevelGRU

        n_weather = _n_base_features()
        n_s1_probs = len(PROBLEM_TYPE_FLAGS)
        input_size = n_weather + n_s1_probs

        model = DangerLevelGRU(input_size=input_size)
        x = torch.randn(8, 7, input_size)
        out = model(x)
        assert out.shape == (8, 4)
        assert input_size == n_weather + 3


class TestBuildS2Sequences:
    def test_vectorized_sequence_shape(self):
        from avalanche_ml.models.gru_stage2 import build_s2_sequences

        n_rows = 50
        n_feat = 10
        n_s1 = 3
        features = np.random.randn(n_rows, n_feat)
        s1_probs = np.random.rand(n_rows, n_s1)
        station_ids = np.array(["S1"] * 25 + ["S2"] * 25)
        lookback = 7

        seqs, labels_idx = build_s2_sequences(
            features, s1_probs, station_ids, lookback,
        )
        assert seqs.ndim == 3
        assert seqs.shape[1] == lookback
        assert seqs.shape[2] == n_feat + n_s1
        expected_n = (25 - lookback) * 2
        assert seqs.shape[0] == expected_n
        assert len(labels_idx) == expected_n

    def test_empty_when_not_enough_history(self):
        from avalanche_ml.models.gru_stage2 import build_s2_sequences

        features = np.random.randn(5, 10)
        s1_probs = np.random.rand(5, 3)
        station_ids = np.array(["S1"] * 5)

        seqs, labels_idx = build_s2_sequences(
            features, s1_probs, station_ids, lookback=7,
        )
        assert seqs.shape[0] == 0
        assert len(labels_idx) == 0
