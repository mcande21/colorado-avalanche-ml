"""RF baseline evaluation notebook.

Runs the full two-stage RF training pipeline, displays evaluation metrics,
confusion matrix, feature importance, and per-elevation-band performance.

Usage:
    python notebooks/02_rf_baseline.py [--parquet path/to/matrix.parquet] [--db path/to/features.db]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from avalanche_ml.features.alignment import PROBLEM_TYPE_FLAGS
from avalanche_ml.features.physics import PHYSICS_FEATURES
from avalanche_ml.features.weather import get_feature_columns
from avalanche_ml.models.evaluation import (
    class_distribution_report,
    danger_confusion_matrix,
    evaluate_stage1,
    evaluate_stage2,
    per_elevation_band_metrics,
)
from avalanche_ml.models.rf_stage2 import TwoStageRFPipeline
from avalanche_ml.models.tracking import configure_tracking, flatten_metrics, log_training_run
from avalanche_ml.models.train import (
    run_training_pipeline_from_dataframe,
    temporal_split_dataframe,
)


def load_data(parquet_path: str | None = None, db_path: str | None = None) -> pd.DataFrame:
    if parquet_path:
        return pd.read_parquet(parquet_path)
    if db_path:
        import duckdb
        conn = duckdb.connect(db_path, read_only=True)
        df = conn.execute("SELECT * FROM training_matrix").fetchdf()
        conn.close()
        return df
    print("ERROR: Provide --parquet or --db path")
    sys.exit(1)


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def display_class_distributions(results: dict) -> None:
    print_section("Class Distributions")
    for split_name, dist in results["class_distributions"].items():
        print(f"\n  {split_name.upper()} (n={dist['total']}):")
        for level in sorted(dist["distribution"].keys()):
            count = dist["distribution"][level]
            pct = dist["percentages"][level]
            bar = "#" * int(pct / 2)
            print(f"    Level {level}: {count:>6} ({pct:>5.1f}%) {bar}")


def display_stage1_metrics(metrics: dict, split_name: str) -> None:
    print(f"\n  Stage 1 — {split_name}:")
    print(f"    Exact match ratio: {metrics['stage1']['exact_match_ratio']:.3f}")
    s1 = metrics["stage1"]["per_type"]
    print(f"    {'Problem Type':<20} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print(f"    {'-' * 40}")
    for pt in PROBLEM_TYPE_FLAGS:
        m = s1[pt]
        print(f"    {pt:<20} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")


def display_stage2_metrics(metrics: dict, split_name: str) -> None:
    print(f"\n  Stage 2 — {split_name}:")
    s2 = metrics["stage2"]
    print(f"    Binary F1:              {s2['binary_f1']:.3f}")
    print(f"    Ordinal accuracy (±1):  {s2['ordinal_accuracy']:.3f}")
    print(f"    Macro F1:               {s2['macro_f1']:.3f}")
    print(f"    Brier score:            {s2['brier_score']:.4f}")
    print(f"    False negative rate:    {s2['false_negative_rate']:.3f}")
    print(f"    High danger detection:  {s2['high_danger_detection_rate']:.3f}")


def display_confusion_matrix(cm: np.ndarray) -> None:
    print_section("Confusion Matrix (Test Set)")
    print(f"    {'':>8}", end="")
    for i in range(1, 6):
        print(f" Pred {i:>2}", end="")
    print()
    for i in range(5):
        print(f"    True {i + 1:>2}", end="")
        for j in range(5):
            print(f" {cm[i, j]:>6}", end="")
        print()


def display_feature_importance(pipeline: TwoStageRFPipeline, top_n: int = 15) -> None:
    print_section(f"Top {top_n} Feature Importance (Stage 2 Binary)")
    importance = pipeline.stage2.global_feature_importance()
    for i, (name, score) in enumerate(importance[:top_n]):
        bar = "#" * int(score * 200)
        print(f"    {i + 1:>2}. {name:<35} {score:.4f} {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RF baseline evaluation")
    parser.add_argument("--parquet", help="Path to training matrix parquet file")
    parser.add_argument("--db", help="Path to DuckDB database")
    parser.add_argument("--output", default="models/rf_baseline", help="Model output directory")
    parser.add_argument("--mlflow-dir", default="mlflow/mlruns", help="MLflow tracking directory")
    args = parser.parse_args()

    print_section("Loading Data")
    data = load_data(args.parquet, args.db)
    print(f"  Loaded {len(data)} rows, {len(data.columns)} columns")

    print_section("Running Training Pipeline")
    results = run_training_pipeline_from_dataframe(
        data,
        mlflow_tracking_dir=args.mlflow_dir,
        model_output_dir=args.output,
    )

    display_class_distributions(results)

    print_section("Evaluation Metrics")
    for split_name, key in [("Train", "train_metrics"), ("Val", "val_metrics"), ("Test", "test_metrics")]:
        display_stage1_metrics(results[key], split_name)
        display_stage2_metrics(results[key], split_name)

    test_s2 = results["test_metrics"]["stage2"]
    if "confusion_matrix" in test_s2:
        display_confusion_matrix(test_s2["confusion_matrix"])

    if results.get("mlflow_run_id"):
        print(f"\n  MLflow run ID: {results['mlflow_run_id']}")

    print_section("Complete")
    print("  Model saved to:", args.output)
    print("  MLflow tracking:", args.mlflow_dir)


if __name__ == "__main__":
    main()
