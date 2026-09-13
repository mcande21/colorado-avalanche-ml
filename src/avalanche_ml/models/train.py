from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from avalanche_ml.features.alignment import PROBLEM_TYPE_FLAGS
from avalanche_ml.features.physics import PHYSICS_FEATURES
from avalanche_ml.features.weather import get_feature_columns
from avalanche_ml.models.evaluation import (
    class_distribution_report,
    evaluate_stage1,
    evaluate_stage2,
)
from avalanche_ml.models.rf_stage2 import TwoStageRFPipeline
from avalanche_ml.models.tracking import configure_tracking, flatten_metrics, log_training_run

TRAIN_END = datetime.date(2023, 6, 30)
VAL_END = datetime.date(2024, 6, 30)


def temporal_split_dataframe(
    df: pd.DataFrame,
    train_end: datetime.date = TRAIN_END,
    val_end: datetime.date = VAL_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(df["date"]).dt.date
    train = df[dates <= train_end].copy()
    val = df[(dates > train_end) & (dates <= val_end)].copy()
    test = df[dates > val_end].copy()
    return train, val, test


def _get_feature_columns() -> list[str]:
    return get_feature_columns() + list(PHYSICS_FEATURES)


def _prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    X = df[feature_cols].copy()
    y_pt = df[PROBLEM_TYPE_FLAGS].copy()
    y_danger = df["danger_level"].values.astype(int)
    return X, y_pt, y_danger


def _evaluate_split(
    pipeline: TwoStageRFPipeline,
    X: pd.DataFrame,
    y_pt: pd.DataFrame,
    y_danger: np.ndarray,
) -> dict:
    stage1_probs = pipeline.stage1.predict(X)
    stage1_pred = (stage1_probs >= 0.5).astype(int)
    stage1_metrics = evaluate_stage1(y_pt, stage1_pred)

    result = pipeline.predict(X)
    stage2_metrics = evaluate_stage2(
        y_danger,
        result["danger_binary"],
        result["danger_binary"] * 2 + 1,
        result["danger_proba"],
    )

    return {"stage1": stage1_metrics, "stage2": stage2_metrics}


def run_training_pipeline_from_dataframe(
    data: pd.DataFrame,
    mlflow_tracking_dir: str | None = None,
    model_output_dir: str | None = None,
) -> dict:
    feature_cols = _get_feature_columns()

    train_df, val_df, test_df = temporal_split_dataframe(data)

    if len(val_df) == 0 and len(test_df) == 0:
        n = len(train_df)
        split1 = int(n * 0.6)
        split2 = int(n * 0.8)
        train_df = data.iloc[:split1].copy()
        val_df = data.iloc[split1:split2].copy()
        test_df = data.iloc[split2:].copy()

    X_train, y_pt_train, y_danger_train = _prepare_xy(train_df, feature_cols)
    X_val, y_pt_val, y_danger_val = _prepare_xy(val_df, feature_cols)
    X_test, y_pt_test, y_danger_test = _prepare_xy(test_df, feature_cols)

    pipeline = TwoStageRFPipeline(
        stage1_kwargs={"n_estimators": 100, "use_smote": True, "random_state": 42},
        stage2_kwargs={"n_estimators": 100, "use_smote": True, "random_state": 42},
    )
    pipeline.train(X_train, y_pt_train, y_danger_train)

    train_metrics = _evaluate_split(pipeline, X_train, y_pt_train, y_danger_train)
    val_metrics = _evaluate_split(pipeline, X_val, y_pt_val, y_danger_val)
    test_metrics = _evaluate_split(pipeline, X_test, y_pt_test, y_danger_test)

    class_dists = {
        "train": class_distribution_report(y_danger_train),
        "val": class_distribution_report(y_danger_val),
        "test": class_distribution_report(y_danger_test),
    }

    mlflow_run_id = None
    if mlflow_tracking_dir:
        configure_tracking(mlflow_tracking_dir)

        flat_metrics = {}
        for split_name, split_metrics in [
            ("train", train_metrics),
            ("val", val_metrics),
            ("test", test_metrics),
        ]:
            for stage_name, stage_metrics in split_metrics.items():
                prefix = f"{split_name}_{stage_name}"
                flattened = flatten_metrics(stage_metrics, prefix)
                for k, v in flattened.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        flat_metrics[k] = v

        params = {
            "n_estimators": 100,
            "use_smote": True,
            "random_state": 42,
        }
        tags = {"model_type": "rf", "stage": "two_stage", "split": "temporal"}

        artifacts = []
        if model_output_dir:
            model_path = Path(model_output_dir)
            pipeline.save(model_path)
            artifacts.append(str(model_path / "stage1" / "models.joblib"))
            artifacts.append(str(model_path / "stage2" / "models.joblib"))

        mlflow_run_id = log_training_run(
            experiment_name="snowpack-instability-rf",
            params=params,
            metrics=flat_metrics,
            tags=tags,
            artifacts=artifacts,
        )
    elif model_output_dir:
        pipeline.save(Path(model_output_dir))

    return {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "class_distributions": class_dists,
        "mlflow_run_id": mlflow_run_id,
    }
