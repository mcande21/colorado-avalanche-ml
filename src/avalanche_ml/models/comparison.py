"""Head-to-head comparison: RF vs LSTM."""
from __future__ import annotations

import mlflow

STAGE2_METRICS = [
    "binary_f1",
    "binary_precision",
    "binary_recall",
    "ordinal_accuracy",
    "macro_f1",
    "brier_score",
    "false_negative_rate",
    "high_danger_detection_rate",
]

LOWER_IS_BETTER = {"brier_score", "false_negative_rate"}

BAND_METRIC_PREFIX = "test_per_band_"


def _extract_stage2_metrics(run) -> dict[str, float]:
    metrics = {}
    for key in STAGE2_METRICS:
        mlflow_key = f"test_stage2_{key}"
        if mlflow_key in run.data.metrics:
            metrics[key] = run.data.metrics[mlflow_key]
    return metrics


def _extract_band_metrics(run) -> dict[str, dict[str, float]]:
    bands: dict[str, dict[str, float]] = {}
    for key, value in run.data.metrics.items():
        if key.startswith(BAND_METRIC_PREFIX):
            rest = key[len(BAND_METRIC_PREFIX):]
            parts = rest.rsplit("_", 1)
            if len(parts) == 2:
                band_name, metric_name = parts
                bands.setdefault(band_name, {})[metric_name] = value
    return bands


def _determine_winner(metric: str, rf_val: float, lstm_val: float) -> str:
    if metric in LOWER_IS_BETTER:
        return "rf" if rf_val <= lstm_val else "lstm"
    return "lstm" if lstm_val >= rf_val else "rf"


def compare_models(rf_run_id: str, lstm_run_id: str) -> dict:
    client = mlflow.tracking.MlflowClient()
    rf_run = client.get_run(rf_run_id)
    lstm_run = client.get_run(lstm_run_id)

    rf_metrics = _extract_stage2_metrics(rf_run)
    lstm_metrics = _extract_stage2_metrics(lstm_run)

    common_keys = set(rf_metrics) & set(lstm_metrics)
    differences = {}
    winner_per_metric = {}
    for key in common_keys:
        differences[key] = lstm_metrics[key] - rf_metrics[key]
        winner_per_metric[key] = _determine_winner(key, rf_metrics[key], lstm_metrics[key])

    rf_bands = _extract_band_metrics(rf_run)
    lstm_bands = _extract_band_metrics(lstm_run)
    per_elevation_band = {}
    for band in set(rf_bands) | set(lstm_bands):
        per_elevation_band[band] = {
            "rf": rf_bands.get(band, {}),
            "lstm": lstm_bands.get(band, {}),
        }

    return {
        "rf_metrics": rf_metrics,
        "lstm_metrics": lstm_metrics,
        "differences": differences,
        "winner_per_metric": winner_per_metric,
        "per_elevation_band": per_elevation_band,
    }


def model_recommendation(comparison_results: dict) -> dict:
    rf = comparison_results["rf_metrics"]
    lstm = comparison_results["lstm_metrics"]

    rf_f1 = rf.get("binary_f1", 0.0)
    lstm_f1 = lstm.get("binary_f1", 0.0)
    rf_fnr = rf.get("false_negative_rate", 1.0)
    lstm_fnr = lstm.get("false_negative_rate", 1.0)

    fnr_threshold = 0.10
    if lstm_fnr - rf_fnr > fnr_threshold:
        return {
            "recommendation": "rf",
            "rationale": (
                f"LSTM false negative rate ({lstm_fnr:.3f}) is significantly worse "
                f"than RF ({rf_fnr:.3f}). In avalanche prediction, missed high-danger "
                f"days are safety-critical. RF recommended despite LSTM binary F1 "
                f"advantage ({lstm_f1:.3f} vs {rf_f1:.3f})."
            ),
        }

    f1_threshold = 0.02
    if lstm_f1 - rf_f1 > f1_threshold:
        return {
            "recommendation": "lstm",
            "rationale": (
                f"LSTM binary F1 ({lstm_f1:.3f}) exceeds RF ({rf_f1:.3f}) by "
                f"{lstm_f1 - rf_f1:.3f}. False negative rates comparable "
                f"(LSTM {lstm_fnr:.3f} vs RF {rf_fnr:.3f})."
            ),
        }

    if rf_f1 - lstm_f1 > f1_threshold:
        return {
            "recommendation": "rf",
            "rationale": (
                f"RF binary F1 ({rf_f1:.3f}) exceeds LSTM ({lstm_f1:.3f}) by "
                f"{rf_f1 - lstm_f1:.3f}."
            ),
        }

    return {
        "recommendation": "rf",
        "rationale": (
            f"Binary F1 within threshold (RF {rf_f1:.3f} vs LSTM {lstm_f1:.3f}). "
            f"RF recommended: simpler, more interpretable, operationally proven."
        ),
    }


def comparison_report(comparison_results: dict) -> str:
    rf = comparison_results["rf_metrics"]
    lstm = comparison_results["lstm_metrics"]
    diffs = comparison_results["differences"]
    winners = comparison_results["winner_per_metric"]

    lines = [
        "=" * 60,
        "MODEL COMPARISON: Random Forest vs Hierarchical LSTM",
        "=" * 60,
        "",
        f"{'Metric':<30} {'RF':>8} {'LSTM':>8} {'Diff':>8} {'Winner':>8}",
        "-" * 60,
    ]

    for key in STAGE2_METRICS:
        rf_val = rf.get(key, float("nan"))
        lstm_val = lstm.get(key, float("nan"))
        diff = diffs.get(key, float("nan"))
        winner = winners.get(key, "n/a")
        lines.append(
            f"{key:<30} {rf_val:>8.4f} {lstm_val:>8.4f} {diff:>+8.4f} {winner:>8}"
        )

    lines.append("")

    band_data = comparison_results.get("per_elevation_band", {})
    if band_data:
        lines.append("Per-Elevation-Band Breakdown:")
        lines.append("-" * 60)
        for band, data in sorted(band_data.items()):
            lines.append(f"  {band}:")
            rf_band = data.get("rf", {})
            lstm_band = data.get("lstm", {})
            for metric in sorted(set(rf_band) | set(lstm_band)):
                rv = rf_band.get(metric, float("nan"))
                lv = lstm_band.get(metric, float("nan"))
                lines.append(f"    {metric:<26} RF={rv:.4f}  LSTM={lv:.4f}")
        lines.append("")

    rec = model_recommendation(comparison_results)
    lines.append("RECOMMENDATION: " + rec["recommendation"].upper())
    lines.append(rec["rationale"])
    lines.append("=" * 60)

    return "\n".join(lines)
