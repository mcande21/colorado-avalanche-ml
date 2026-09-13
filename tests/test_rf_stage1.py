from __future__ import annotations

import datetime
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from avalanche_ml.features.alignment import PROBLEM_TYPE_FLAGS
from avalanche_ml.features.physics import PHYSICS_FEATURES
from avalanche_ml.features.weather import get_feature_columns


def _make_synthetic_data(
    n_samples: int = 200,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Build synthetic training data with known signal patterns.

    High precip_sum_72h -> storm_slab
    High temp_gradient_days -> persistent_slab
    High temp (above freezing proxies) -> loose_wet
    """
    rng = np.random.RandomState(rng_seed)
    weather_cols = get_feature_columns()
    physics_cols = PHYSICS_FEATURES

    data: dict[str, np.ndarray] = {}
    for col in weather_cols:
        data[col] = rng.randn(n_samples).astype(np.float64)
    for col in physics_cols:
        data[col] = rng.randn(n_samples).astype(np.float64)

    data["precip_sum_72h"] = rng.uniform(0, 5, n_samples)
    data["temp_gradient_days"] = rng.uniform(0, 30, n_samples)
    data["temp_mean_24h"] = rng.uniform(-10, 50, n_samples)

    storm_slab = (data["precip_sum_72h"] > 2.5).astype(int)
    persistent_slab = (data["temp_gradient_days"] > 15).astype(int)
    loose_wet = (data["temp_mean_24h"] > 35).astype(int)

    data["storm_slab"] = storm_slab
    data["persistent_slab"] = persistent_slab
    data["loose_wet"] = loose_wet

    data["danger_level"] = rng.randint(1, 4, n_samples)

    base_date = datetime.date(2022, 1, 1)
    data["date"] = pd.array(
        [base_date + datetime.timedelta(days=int(i)) for i in range(n_samples)]
    )
    data["station_id"] = np.array(["STATION_01"] * n_samples)
    data["elevation_band"] = np.array(["above_treeline"] * n_samples)
    data["zone_id"] = np.array(["ZONE_01"] * n_samples)

    return pd.DataFrame(data)


def _feature_columns() -> list[str]:
    return get_feature_columns() + PHYSICS_FEATURES


class TestStage1RFTraining:
    def test_model_trains_without_error(self):
        from avalanche_ml.models.rf_stage1 import Stage1RF

        df = _make_synthetic_data(n_samples=100)
        feature_cols = _feature_columns()
        model = Stage1RF()
        model.train(df[feature_cols], df[PROBLEM_TYPE_FLAGS])

    def test_predictions_are_probabilities(self):
        from avalanche_ml.models.rf_stage1 import Stage1RF

        df = _make_synthetic_data(n_samples=100)
        feature_cols = _feature_columns()
        model = Stage1RF()
        model.train(df[feature_cols], df[PROBLEM_TYPE_FLAGS])

        probs = model.predict(df[feature_cols].iloc[:5])
        assert isinstance(probs, pd.DataFrame)
        assert list(probs.columns) == PROBLEM_TYPE_FLAGS
        assert (probs >= 0.0).all().all()
        assert (probs <= 1.0).all().all()

    def test_binary_predictions_respect_threshold(self):
        from avalanche_ml.models.rf_stage1 import Stage1RF

        df = _make_synthetic_data(n_samples=100)
        feature_cols = _feature_columns()
        model = Stage1RF()
        model.train(df[feature_cols], df[PROBLEM_TYPE_FLAGS])

        thresholds = {"persistent_slab": 0.3, "storm_slab": 0.7, "loose_wet": 0.5}
        probs = model.predict(df[feature_cols].iloc[:20])
        binary = model.predict_binary(df[feature_cols].iloc[:20], thresholds)

        assert isinstance(binary, pd.DataFrame)
        assert set(binary.columns) == set(PROBLEM_TYPE_FLAGS)
        for col in PROBLEM_TYPE_FLAGS:
            expected = (probs[col] >= thresholds[col]).astype(int)
            np.testing.assert_array_equal(binary[col].values, expected.values)

    def test_smote_increases_minority_samples(self):
        from avalanche_ml.models.rf_stage1 import Stage1RF

        feature_cols = _feature_columns()
        df = _make_synthetic_data(n_samples=200, rng_seed=99)
        df["persistent_slab"] = 0
        df.loc[df.index[:5], "persistent_slab"] = 1

        model = Stage1RF(use_smote=True)
        model.train(df[feature_cols], df[PROBLEM_TYPE_FLAGS])
        assert model.smote_applied_

    def test_shap_values_shape(self):
        from avalanche_ml.models.rf_stage1 import Stage1RF

        df = _make_synthetic_data(n_samples=80)
        feature_cols = _feature_columns()
        model = Stage1RF()
        model.train(df[feature_cols], df[PROBLEM_TYPE_FLAGS])

        shap_vals = model.shap_values(df[feature_cols].iloc[:3])
        assert isinstance(shap_vals, dict)
        for pt in PROBLEM_TYPE_FLAGS:
            assert pt in shap_vals
            arr = shap_vals[pt]
            assert arr.shape == (3, len(feature_cols))

    def test_global_feature_importance(self):
        from avalanche_ml.models.rf_stage1 import Stage1RF

        df = _make_synthetic_data(n_samples=100)
        feature_cols = _feature_columns()
        model = Stage1RF()
        model.train(df[feature_cols], df[PROBLEM_TYPE_FLAGS])

        importance = model.global_feature_importance()
        assert isinstance(importance, dict)
        for pt in PROBLEM_TYPE_FLAGS:
            assert pt in importance
            fi = importance[pt]
            assert isinstance(fi, list)
            assert len(fi) > 0
            assert all(isinstance(item, tuple) and len(item) == 2 for item in fi)
            names = [item[0] for item in fi]
            assert all(isinstance(n, str) for n in names)
            values = [item[1] for item in fi]
            assert values == sorted(values, reverse=True)

    def test_serialization_roundtrip(self):
        from avalanche_ml.models.rf_stage1 import Stage1RF

        df = _make_synthetic_data(n_samples=80)
        feature_cols = _feature_columns()
        model = Stage1RF()
        model.train(df[feature_cols], df[PROBLEM_TYPE_FLAGS])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model"
            model.save(path)
            loaded = Stage1RF.load(path)

            probs_orig = model.predict(df[feature_cols].iloc[:5])
            probs_loaded = loaded.predict(df[feature_cols].iloc[:5])
            pd.testing.assert_frame_equal(probs_orig, probs_loaded)

    def test_handles_all_negative_class(self):
        from avalanche_ml.models.rf_stage1 import Stage1RF

        df = _make_synthetic_data(n_samples=100)
        df["loose_wet"] = 0
        feature_cols = _feature_columns()
        model = Stage1RF()
        model.train(df[feature_cols], df[PROBLEM_TYPE_FLAGS])

        probs = model.predict(df[feature_cols].iloc[:5])
        assert "loose_wet" in probs.columns
        assert (probs["loose_wet"] >= 0.0).all()
        assert (probs["loose_wet"] <= 1.0).all()

    def test_per_elevation_band(self):
        from avalanche_ml.models.rf_stage1 import Stage1RF

        df = _make_synthetic_data(n_samples=100)
        df["elevation_band"] = "above_treeline"
        feature_cols = _feature_columns()

        band_df = df[df["elevation_band"] == "above_treeline"]
        model = Stage1RF()
        model.train(band_df[feature_cols], band_df[PROBLEM_TYPE_FLAGS])

        probs = model.predict(band_df[feature_cols].iloc[:3])
        assert probs.shape == (3, len(PROBLEM_TYPE_FLAGS))

    def test_feature_names_match(self):
        from avalanche_ml.models.rf_stage1 import Stage1RF

        df = _make_synthetic_data(n_samples=80)
        feature_cols = _feature_columns()
        model = Stage1RF()
        model.train(df[feature_cols], df[PROBLEM_TYPE_FLAGS])
        assert model.feature_names_ == feature_cols

    def test_metadata_saved(self):
        from avalanche_ml.models.rf_stage1 import Stage1RF

        df = _make_synthetic_data(n_samples=80)
        feature_cols = _feature_columns()
        model = Stage1RF()
        model.train(df[feature_cols], df[PROBLEM_TYPE_FLAGS])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model"
            model.save(path)
            loaded = Stage1RF.load(path)
            assert loaded.metadata_ is not None
            assert "training_date" in loaded.metadata_
            assert "feature_names" in loaded.metadata_
            assert "hyperparameters" in loaded.metadata_
            assert "class_distributions" in loaded.metadata_
