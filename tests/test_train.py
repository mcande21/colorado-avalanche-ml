from __future__ import annotations

import numpy as np
import pandas as pd

from avalanche_ml.features.physics import PHYSICS_FEATURES
from avalanche_ml.features.weather import get_feature_columns

ELEVATION_BANDS = ["above_treeline", "near_treeline", "below_treeline"]


def _make_synthetic_training_data(n_per_band: int = 60, rng_seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(rng_seed)
    weather_cols = get_feature_columns()
    physics_cols = PHYSICS_FEATURES
    rows = []

    for band in ELEVATION_BANDS:
        n = n_per_band
        data: dict[str, np.ndarray] = {}
        for col in weather_cols:
            data[col] = rng.randn(n).astype(np.float64)
        for col in physics_cols:
            data[col] = rng.randn(n).astype(np.float64)

        data["precip_sum_72h"] = rng.uniform(0, 5, n)
        data["temp_gradient_days"] = rng.uniform(0, 30, n)
        data["temp_mean_24h"] = rng.uniform(-10, 50, n)

        danger = np.ones(n, dtype=int)
        danger[data["precip_sum_72h"] > 3.0] = 3
        danger[data["temp_gradient_days"] > 20] = 4
        danger[(data["precip_sum_72h"] > 4.0) & (data["temp_gradient_days"] > 25)] = 5
        moderate_mask = (data["precip_sum_72h"] > 2.0) & (danger == 1)
        danger[moderate_mask] = 2
        data["danger_level"] = danger

        data["persistent_slab"] = (data["temp_gradient_days"] > 15).astype(int)
        data["storm_slab"] = (data["precip_sum_72h"] > 2.5).astype(int)
        data["loose_wet"] = (data["temp_mean_24h"] > 35).astype(int)

        data["elevation_band"] = np.array([band] * n)
        data["zone_id"] = np.array(["zone_01"] * n)
        data["station_id"] = np.array(["stn_01"] * n)

        import datetime
        base = datetime.date(2022, 1, 1)
        step = datetime.timedelta(days=30)
        data["date"] = pd.array(
            [base + step * i for i in range(n)],
            dtype="object",
        )

        df = pd.DataFrame(data)
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


class TestTrainPipeline:
    def test_run_training_pipeline(self, tmp_path):
        from avalanche_ml.models.train import run_training_pipeline_from_dataframe

        data = _make_synthetic_training_data()
        results = run_training_pipeline_from_dataframe(
            data,
            mlflow_tracking_dir=str(tmp_path / "mlruns"),
            model_output_dir=str(tmp_path / "models"),
        )

        assert "train_metrics" in results
        assert "val_metrics" in results
        assert "test_metrics" in results
        assert "mlflow_run_id" in results

        for split in ["train_metrics", "val_metrics", "test_metrics"]:
            m = results[split]
            assert "stage2" in m
            assert "binary_f1" in m["stage2"]
            assert "ordinal_accuracy" in m["stage2"]

    def test_model_artifacts_saved(self, tmp_path):
        from avalanche_ml.models.train import run_training_pipeline_from_dataframe

        data = _make_synthetic_training_data()
        run_training_pipeline_from_dataframe(
            data,
            mlflow_tracking_dir=str(tmp_path / "mlruns"),
            model_output_dir=str(tmp_path / "models"),
        )

        model_dir = tmp_path / "models"
        assert (model_dir / "stage1" / "models.joblib").exists()
        assert (model_dir / "stage2" / "models.joblib").exists()

    def test_temporal_split_from_dataframe(self):
        from avalanche_ml.models.train import temporal_split_dataframe

        data = _make_synthetic_training_data()
        train, val, test = temporal_split_dataframe(data)

        assert len(train) > 0
        assert len(val) > 0 or len(test) > 0

    def test_class_distribution_in_results(self, tmp_path):
        from avalanche_ml.models.train import run_training_pipeline_from_dataframe

        data = _make_synthetic_training_data()
        results = run_training_pipeline_from_dataframe(
            data,
            mlflow_tracking_dir=str(tmp_path / "mlruns"),
            model_output_dir=str(tmp_path / "models"),
        )
        assert "class_distributions" in results
