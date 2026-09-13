"""Tests for model comparison (RF vs LSTM)."""
from __future__ import annotations


def _make_comparison_results(
    rf_binary_f1: float = 0.75,
    lstm_binary_f1: float = 0.80,
    rf_fnr: float = 0.15,
    lstm_fnr: float = 0.10,
) -> dict:
    return {
        "rf_metrics": {
            "binary_f1": rf_binary_f1,
            "binary_precision": 0.70,
            "binary_recall": 0.80,
            "ordinal_accuracy": 0.65,
            "macro_f1": 0.60,
            "brier_score": 0.20,
            "false_negative_rate": rf_fnr,
            "high_danger_detection_rate": 0.90,
        },
        "lstm_metrics": {
            "binary_f1": lstm_binary_f1,
            "binary_precision": 0.75,
            "binary_recall": 0.85,
            "ordinal_accuracy": 0.70,
            "macro_f1": 0.65,
            "brier_score": 0.18,
            "false_negative_rate": lstm_fnr,
            "high_danger_detection_rate": 0.95,
        },
        "differences": {
            "binary_f1": lstm_binary_f1 - rf_binary_f1,
            "binary_precision": 0.05,
            "binary_recall": 0.05,
            "ordinal_accuracy": 0.05,
            "macro_f1": 0.05,
            "brier_score": -0.02,
            "false_negative_rate": lstm_fnr - rf_fnr,
            "high_danger_detection_rate": 0.05,
        },
        "winner_per_metric": {
            "binary_f1": "lstm",
            "binary_precision": "lstm",
            "binary_recall": "lstm",
            "ordinal_accuracy": "lstm",
            "macro_f1": "lstm",
            "brier_score": "lstm",
            "false_negative_rate": "lstm",
            "high_danger_detection_rate": "lstm",
        },
        "per_elevation_band": {},
    }


class TestCompareModels:
    def test_compare_returns_differences(self, tmp_path):
        import mlflow
        from avalanche_ml.models.tracking import configure_tracking

        configure_tracking(str(tmp_path / "mlruns"))

        mlflow.set_experiment("snowpack-instability-rf")
        with mlflow.start_run() as rf_run:
            mlflow.log_metric("test_stage2_binary_f1", 0.75)
            mlflow.log_metric("test_stage2_binary_precision", 0.70)
            mlflow.log_metric("test_stage2_binary_recall", 0.80)
            mlflow.log_metric("test_stage2_ordinal_accuracy", 0.65)
            mlflow.log_metric("test_stage2_macro_f1", 0.60)
            mlflow.log_metric("test_stage2_brier_score", 0.20)
            mlflow.log_metric("test_stage2_false_negative_rate", 0.15)
            mlflow.log_metric("test_stage2_high_danger_detection_rate", 0.90)
            rf_run_id = rf_run.info.run_id

        mlflow.set_experiment("snowpack-instability-lstm")
        with mlflow.start_run() as lstm_run:
            mlflow.log_metric("test_stage2_binary_f1", 0.80)
            mlflow.log_metric("test_stage2_binary_precision", 0.75)
            mlflow.log_metric("test_stage2_binary_recall", 0.85)
            mlflow.log_metric("test_stage2_ordinal_accuracy", 0.70)
            mlflow.log_metric("test_stage2_macro_f1", 0.65)
            mlflow.log_metric("test_stage2_brier_score", 0.18)
            mlflow.log_metric("test_stage2_false_negative_rate", 0.10)
            mlflow.log_metric("test_stage2_high_danger_detection_rate", 0.95)
            lstm_run_id = lstm_run.info.run_id

        from avalanche_ml.models.comparison import compare_models

        result = compare_models(rf_run_id, lstm_run_id)
        assert "rf_metrics" in result
        assert "lstm_metrics" in result
        assert "differences" in result
        assert "winner_per_metric" in result

    def test_winner_per_metric_correct(self, tmp_path):
        import mlflow
        from avalanche_ml.models.tracking import configure_tracking

        configure_tracking(str(tmp_path / "mlruns"))

        mlflow.set_experiment("snowpack-instability-rf")
        with mlflow.start_run() as rf_run:
            mlflow.log_metric("test_stage2_binary_f1", 0.80)
            mlflow.log_metric("test_stage2_false_negative_rate", 0.10)
            mlflow.log_metric("test_stage2_brier_score", 0.15)
            rf_run_id = rf_run.info.run_id

        mlflow.set_experiment("snowpack-instability-lstm")
        with mlflow.start_run() as lstm_run:
            mlflow.log_metric("test_stage2_binary_f1", 0.70)
            mlflow.log_metric("test_stage2_false_negative_rate", 0.20)
            mlflow.log_metric("test_stage2_brier_score", 0.25)
            lstm_run_id = lstm_run.info.run_id

        from avalanche_ml.models.comparison import compare_models

        result = compare_models(rf_run_id, lstm_run_id)
        assert result["winner_per_metric"]["binary_f1"] == "rf"
        assert result["winner_per_metric"]["false_negative_rate"] == "rf"
        assert result["winner_per_metric"]["brier_score"] == "rf"

    def test_fnr_direction_correct(self, tmp_path):
        """Lower FNR is better - check that direction is handled correctly."""
        import mlflow
        from avalanche_ml.models.tracking import configure_tracking

        configure_tracking(str(tmp_path / "mlruns"))

        mlflow.set_experiment("snowpack-instability-rf")
        with mlflow.start_run() as rf_run:
            mlflow.log_metric("test_stage2_binary_f1", 0.75)
            mlflow.log_metric("test_stage2_false_negative_rate", 0.05)
            rf_run_id = rf_run.info.run_id

        mlflow.set_experiment("snowpack-instability-lstm")
        with mlflow.start_run() as lstm_run:
            mlflow.log_metric("test_stage2_binary_f1", 0.75)
            mlflow.log_metric("test_stage2_false_negative_rate", 0.20)
            lstm_run_id = lstm_run.info.run_id

        from avalanche_ml.models.comparison import compare_models

        result = compare_models(rf_run_id, lstm_run_id)
        assert result["winner_per_metric"]["false_negative_rate"] == "rf"


class TestModelRecommendation:
    def test_lstm_wins_when_f1_higher_by_threshold(self):
        from avalanche_ml.models.comparison import model_recommendation

        comparison = _make_comparison_results(
            rf_binary_f1=0.72, lstm_binary_f1=0.78,
            rf_fnr=0.15, lstm_fnr=0.10,
        )
        rec = model_recommendation(comparison)
        assert rec["recommendation"] == "lstm"
        assert "rationale" in rec

    def test_rf_wins_when_f1_higher_by_threshold(self):
        from avalanche_ml.models.comparison import model_recommendation

        comparison = _make_comparison_results(
            rf_binary_f1=0.80, lstm_binary_f1=0.75,
            rf_fnr=0.10, lstm_fnr=0.15,
        )
        rec = model_recommendation(comparison)
        assert rec["recommendation"] == "rf"

    def test_tie_defaults_to_rf(self):
        from avalanche_ml.models.comparison import model_recommendation

        comparison = _make_comparison_results(
            rf_binary_f1=0.76, lstm_binary_f1=0.77,
            rf_fnr=0.12, lstm_fnr=0.12,
        )
        rec = model_recommendation(comparison)
        assert rec["recommendation"] == "rf"

    def test_lstm_fnr_much_worse_overrides(self):
        """Even if LSTM binary F1 is better, high FNR should flag concern."""
        from avalanche_ml.models.comparison import model_recommendation

        comparison = _make_comparison_results(
            rf_binary_f1=0.72, lstm_binary_f1=0.78,
            rf_fnr=0.05, lstm_fnr=0.25,
        )
        rec = model_recommendation(comparison)
        assert rec["recommendation"] == "rf"
        assert "false negative" in rec["rationale"].lower()


class TestComparisonReport:
    def test_report_is_string(self):
        from avalanche_ml.models.comparison import comparison_report

        comparison = _make_comparison_results()
        report = comparison_report(comparison)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_contains_metrics(self):
        from avalanche_ml.models.comparison import comparison_report

        comparison = _make_comparison_results()
        report = comparison_report(comparison)
        assert "binary_f1" in report.lower() or "Binary F1" in report

    def test_report_contains_recommendation(self):
        from avalanche_ml.models.comparison import comparison_report

        comparison = _make_comparison_results()
        report = comparison_report(comparison)
        lower = report.lower()
        assert "recommend" in lower or "winner" in lower


class TestPerElevationBandComparison:
    def test_per_band_in_comparison(self, tmp_path):
        import mlflow
        from avalanche_ml.models.tracking import configure_tracking

        configure_tracking(str(tmp_path / "mlruns"))

        mlflow.set_experiment("snowpack-instability-rf")
        with mlflow.start_run() as rf_run:
            mlflow.log_metric("test_stage2_binary_f1", 0.75)
            mlflow.log_metric("test_per_band_above_treeline_macro_f1", 0.70)
            mlflow.log_metric("test_per_band_below_treeline_macro_f1", 0.60)
            rf_run_id = rf_run.info.run_id

        mlflow.set_experiment("snowpack-instability-lstm")
        with mlflow.start_run() as lstm_run:
            mlflow.log_metric("test_stage2_binary_f1", 0.80)
            mlflow.log_metric("test_per_band_above_treeline_macro_f1", 0.75)
            mlflow.log_metric("test_per_band_below_treeline_macro_f1", 0.65)
            lstm_run_id = lstm_run.info.run_id

        from avalanche_ml.models.comparison import compare_models

        result = compare_models(rf_run_id, lstm_run_id)
        assert "per_elevation_band" in result
        if result["per_elevation_band"]:
            for band_data in result["per_elevation_band"].values():
                assert "rf" in band_data
                assert "lstm" in band_data
