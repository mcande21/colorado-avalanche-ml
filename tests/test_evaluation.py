from __future__ import annotations

import numpy as np
import pandas as pd

from avalanche_ml.features.alignment import PROBLEM_TYPE_FLAGS

ELEVATION_BANDS = ["above_treeline", "near_treeline", "below_treeline"]


class TestBinaryMetrics:
    def test_binary_f1_perfect(self):
        from avalanche_ml.models.evaluation import compute_binary_metrics

        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        metrics = compute_binary_metrics(y_true, y_pred)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_binary_f1_imperfect(self):
        from avalanche_ml.models.evaluation import compute_binary_metrics

        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 0])
        metrics = compute_binary_metrics(y_true, y_pred)
        # TP=2, FP=1, FN=1 -> precision=2/3, recall=2/3, f1=2/3
        assert abs(metrics["precision"] - 2 / 3) < 1e-6
        assert abs(metrics["recall"] - 2 / 3) < 1e-6
        assert abs(metrics["f1"] - 2 / 3) < 1e-6

    def test_binary_f1_all_negative(self):
        from avalanche_ml.models.evaluation import compute_binary_metrics

        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        metrics = compute_binary_metrics(y_true, y_pred)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0


class TestOrdinalAccuracy:
    def test_ordinal_exact_match(self):
        from avalanche_ml.models.evaluation import ordinal_accuracy

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert ordinal_accuracy(y_true, y_pred) == 1.0

    def test_ordinal_within_one(self):
        from avalanche_ml.models.evaluation import ordinal_accuracy

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 4])
        assert ordinal_accuracy(y_true, y_pred) == 1.0

    def test_ordinal_some_outside(self):
        from avalanche_ml.models.evaluation import ordinal_accuracy

        y_true = np.array([1, 2, 5, 4])
        y_pred = np.array([1, 4, 3, 4])
        # diffs: 0, 2, 2, 0 -> within ±1: 2/4 = 0.5
        assert ordinal_accuracy(y_true, y_pred) == 0.5


class TestConfusionMatrix:
    def test_confusion_matrix_shape(self):
        from avalanche_ml.models.evaluation import danger_confusion_matrix

        y_true = np.array([1, 2, 3, 4, 5, 1, 2])
        y_pred = np.array([1, 2, 3, 4, 5, 2, 1])
        cm = danger_confusion_matrix(y_true, y_pred)
        assert cm.shape == (5, 5)

    def test_confusion_matrix_values(self):
        from avalanche_ml.models.evaluation import danger_confusion_matrix

        y_true = np.array([1, 1, 2, 2, 3])
        y_pred = np.array([1, 2, 2, 3, 3])
        cm = danger_confusion_matrix(y_true, y_pred)
        assert cm[0, 0] == 1  # true=1, pred=1
        assert cm[0, 1] == 1  # true=1, pred=2
        assert cm[1, 1] == 1  # true=2, pred=2
        assert cm[1, 2] == 1  # true=2, pred=3
        assert cm[2, 2] == 1  # true=3, pred=3


class TestFalseNegativeRate:
    def test_false_negative_rate_perfect(self):
        from avalanche_ml.models.evaluation import false_negative_rate_elevated

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred_binary = np.array([0, 0, 1, 1, 1])
        assert false_negative_rate_elevated(y_true, y_pred_binary) == 0.0

    def test_false_negative_rate_all_missed(self):
        from avalanche_ml.models.evaluation import false_negative_rate_elevated

        y_true = np.array([3, 4, 5])
        y_pred_binary = np.array([0, 0, 0])
        assert false_negative_rate_elevated(y_true, y_pred_binary) == 1.0

    def test_false_negative_rate_partial(self):
        from avalanche_ml.models.evaluation import false_negative_rate_elevated

        y_true = np.array([1, 2, 3, 4, 5, 3])
        y_pred_binary = np.array([0, 0, 1, 0, 1, 0])
        # elevated: indices 2,3,4,5 (danger>=3). pred_binary: 1,0,1,0
        # FN = 2 out of 4 -> 0.5
        assert false_negative_rate_elevated(y_true, y_pred_binary) == 0.5

    def test_false_negative_rate_no_elevated(self):
        from avalanche_ml.models.evaluation import false_negative_rate_elevated

        y_true = np.array([1, 2, 1])
        y_pred_binary = np.array([0, 0, 0])
        assert false_negative_rate_elevated(y_true, y_pred_binary) == 0.0


class TestHighDangerDetection:
    def test_detection_rate_perfect(self):
        from avalanche_ml.models.evaluation import high_danger_detection_rate

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred_binary = np.array([0, 0, 0, 1, 1])
        assert high_danger_detection_rate(y_true, y_pred_binary) == 1.0

    def test_detection_rate_missed(self):
        from avalanche_ml.models.evaluation import high_danger_detection_rate

        y_true = np.array([4, 5, 4, 5])
        y_pred_binary = np.array([0, 1, 0, 1])
        assert high_danger_detection_rate(y_true, y_pred_binary) == 0.5

    def test_detection_rate_no_high(self):
        from avalanche_ml.models.evaluation import high_danger_detection_rate

        y_true = np.array([1, 2, 3])
        y_pred_binary = np.array([0, 0, 1])
        assert high_danger_detection_rate(y_true, y_pred_binary) == 1.0


class TestBrierScore:
    def test_brier_score_perfect(self):
        from avalanche_ml.models.evaluation import brier_score

        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.0, 1.0, 1.0, 0.0])
        assert brier_score(y_true, y_prob) == 0.0

    def test_brier_score_worst(self):
        from avalanche_ml.models.evaluation import brier_score

        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([1.0, 0.0, 0.0, 1.0])
        assert brier_score(y_true, y_prob) == 1.0

    def test_brier_score_value(self):
        from avalanche_ml.models.evaluation import brier_score

        y_true = np.array([1, 0])
        y_prob = np.array([0.8, 0.2])
        # (1-0.8)^2 + (0-0.2)^2 = 0.04 + 0.04 = 0.08 / 2 = 0.04
        assert abs(brier_score(y_true, y_prob) - 0.04) < 1e-6


class TestClassDistribution:
    def test_class_distribution_report(self):
        from avalanche_ml.models.evaluation import class_distribution_report

        y = np.array([1, 1, 1, 2, 2, 3])
        report = class_distribution_report(y)
        assert report["total"] == 6
        assert report["distribution"][1] == 3
        assert report["distribution"][2] == 2
        assert report["distribution"][3] == 1
        assert abs(report["percentages"][1] - 50.0) < 1e-6


class TestPerElevationBandMetrics:
    def test_per_band_breakdown(self):
        from avalanche_ml.models.evaluation import per_elevation_band_metrics

        y_true = np.array([1, 2, 3, 1, 2, 3])
        y_pred = np.array([1, 2, 3, 2, 2, 3])
        bands = np.array([
            "above_treeline", "above_treeline",
            "near_treeline", "near_treeline",
            "below_treeline", "below_treeline",
        ])
        result = per_elevation_band_metrics(y_true, y_pred, bands)
        assert "above_treeline" in result
        assert "near_treeline" in result
        assert "below_treeline" in result
        for band_metrics in result.values():
            assert "ordinal_accuracy" in band_metrics
            assert "macro_f1" in band_metrics


class TestStage1Evaluation:
    def test_stage1_metrics(self):
        from avalanche_ml.models.evaluation import evaluate_stage1

        y_true = pd.DataFrame({
            "persistent_slab": [1, 0, 1, 0, 1],
            "storm_slab": [0, 1, 0, 1, 0],
            "loose_wet": [0, 0, 1, 1, 0],
        })
        y_pred = pd.DataFrame({
            "persistent_slab": [1, 0, 1, 1, 1],
            "storm_slab": [0, 1, 1, 1, 0],
            "loose_wet": [0, 0, 1, 0, 0],
        })
        metrics = evaluate_stage1(y_true, y_pred)
        assert "per_type" in metrics
        for pt in PROBLEM_TYPE_FLAGS:
            assert pt in metrics["per_type"]
            assert "precision" in metrics["per_type"][pt]
            assert "recall" in metrics["per_type"][pt]
            assert "f1" in metrics["per_type"][pt]
        assert "exact_match_ratio" in metrics

    def test_exact_match_ratio(self):
        from avalanche_ml.models.evaluation import evaluate_stage1

        y_true = pd.DataFrame({
            "persistent_slab": [1, 0, 1],
            "storm_slab": [0, 1, 0],
            "loose_wet": [0, 0, 1],
        })
        y_pred = pd.DataFrame({
            "persistent_slab": [1, 0, 0],
            "storm_slab": [0, 1, 0],
            "loose_wet": [0, 0, 0],
        })
        metrics = evaluate_stage1(y_true, y_pred)
        # Row 0: match, Row 1: match, Row 2: mismatch -> 2/3
        assert abs(metrics["exact_match_ratio"] - 2 / 3) < 1e-6


class TestStage2Evaluation:
    def test_stage2_metrics(self):
        from avalanche_ml.models.evaluation import evaluate_stage2

        y_true = np.array([1, 2, 3, 4, 5, 2, 3, 1])
        y_pred_binary = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred_danger = np.array([1, 2, 3, 4, 5, 2, 3, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.7, 0.9, 0.95, 0.15, 0.8, 0.05])
        metrics = evaluate_stage2(y_true, y_pred_binary, y_pred_danger, y_pred_proba)

        assert "binary_f1" in metrics
        assert "ordinal_accuracy" in metrics
        assert "confusion_matrix" in metrics
        assert "macro_f1" in metrics
        assert "brier_score" in metrics
        assert "false_negative_rate" in metrics
        assert "high_danger_detection_rate" in metrics

    def test_stage2_macro_f1(self):
        from avalanche_ml.models.evaluation import evaluate_stage2

        y_true = np.array([1, 1, 2, 2, 3, 3])
        y_pred_binary = np.array([0, 0, 0, 0, 1, 1])
        y_pred_danger = np.array([1, 1, 2, 2, 3, 3])
        y_pred_proba = np.array([0.1, 0.1, 0.3, 0.3, 0.8, 0.8])
        metrics = evaluate_stage2(y_true, y_pred_binary, y_pred_danger, y_pred_proba)
        assert metrics["macro_f1"] == 1.0
