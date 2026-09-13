from __future__ import annotations

import datetime
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from avalanche_ml.features.alignment import PROBLEM_TYPE_FLAGS
from avalanche_ml.features.physics import PHYSICS_FEATURES
from avalanche_ml.features.weather import get_feature_columns

ELEVATION_BANDS = ["above_treeline", "near_treeline", "below_treeline"]
DANGER_LEVELS = [1, 2, 3, 4, 5]


def _make_synthetic_stage2_data(
    n_samples: int = 300,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Synthetic data where high persistent_slab_prob + high temp_gradient_days -> elevated danger."""
    rng = np.random.RandomState(rng_seed)
    weather_cols = get_feature_columns()
    physics_cols = PHYSICS_FEATURES

    data: dict[str, np.ndarray] = {}
    for col in weather_cols:
        data[col] = rng.randn(n_samples).astype(np.float64)
    for col in physics_cols:
        data[col] = rng.randn(n_samples).astype(np.float64)

    data["temp_gradient_days"] = rng.uniform(0, 30, n_samples)

    data["persistent_slab_prob"] = rng.uniform(0, 1, n_samples)
    data["storm_slab_prob"] = rng.uniform(0, 1, n_samples)
    data["loose_wet_prob"] = rng.uniform(0, 1, n_samples)

    danger = np.ones(n_samples, dtype=int)
    elevated_mask = (
        (data["persistent_slab_prob"] > 0.6) & (data["temp_gradient_days"] > 15)
    )
    danger[elevated_mask] = rng.choice([3, 4, 5], size=elevated_mask.sum())
    moderate_mask = ~elevated_mask & (data["persistent_slab_prob"] > 0.4)
    danger[moderate_mask] = 2
    data["danger_level"] = danger

    data["elevation_band"] = np.array(
        [ELEVATION_BANDS[i % 3] for i in range(n_samples)]
    )

    base_date = datetime.date(2022, 1, 1)
    data["date"] = pd.array(
        [base_date + datetime.timedelta(days=int(i)) for i in range(n_samples)]
    )
    data["station_id"] = np.array(["STATION_01"] * n_samples)
    data["zone_id"] = np.array(["ZONE_01"] * n_samples)

    for pt in PROBLEM_TYPE_FLAGS:
        data[pt] = (rng.rand(n_samples) > 0.7).astype(int)

    return pd.DataFrame(data)


def _feature_columns() -> list[str]:
    return get_feature_columns() + PHYSICS_FEATURES


def _stage2_feature_columns() -> list[str]:
    return _feature_columns() + [
        "persistent_slab_prob", "storm_slab_prob", "loose_wet_prob",
    ]


class TestStage2RFTraining:
    def test_model_trains_without_error(self):
        from avalanche_ml.models.rf_stage2 import Stage2RF

        df = _make_synthetic_stage2_data(n_samples=200)
        feature_cols = _stage2_feature_columns()
        model = Stage2RF()
        model.train(df[feature_cols], df["danger_level"])

    def test_binary_prediction_elevated_vs_not(self):
        from avalanche_ml.models.rf_stage2 import Stage2RF

        df = _make_synthetic_stage2_data(n_samples=200)
        feature_cols = _stage2_feature_columns()
        model = Stage2RF()
        model.train(df[feature_cols], df["danger_level"])

        result = model.predict(df[feature_cols].iloc[:10])
        assert "danger_binary" in result
        binary = result["danger_binary"]
        assert len(binary) == 10
        assert all(v in (0, 1) for v in binary)

    def test_probability_distribution_sums_to_one(self):
        from avalanche_ml.models.rf_stage2 import Stage2RF

        df = _make_synthetic_stage2_data(n_samples=200)
        feature_cols = _stage2_feature_columns()
        model = Stage2RF()
        model.train(df[feature_cols], df["danger_level"])

        result = model.predict(df[feature_cols].iloc[:10])
        assert "danger_distribution" in result
        dist = result["danger_distribution"]
        assert dist.shape[0] == 10
        row_sums = dist.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=0.01)

    def test_smote_applied_for_binary_target(self):
        from avalanche_ml.models.rf_stage2 import Stage2RF

        df = _make_synthetic_stage2_data(n_samples=200)
        feature_cols = _stage2_feature_columns()
        model = Stage2RF(use_smote=True)
        model.train(df[feature_cols], df["danger_level"])
        assert model.smote_applied_

    def test_asymmetric_cost_higher_penalty_false_negatives(self):
        from avalanche_ml.models.rf_stage2 import Stage2RF

        model = Stage2RF()
        cost_matrix = model.get_cost_matrix()
        assert cost_matrix[1][0] > cost_matrix[0][1]

    def test_shap_values_include_problem_type_features(self):
        from avalanche_ml.models.rf_stage2 import Stage2RF

        df = _make_synthetic_stage2_data(n_samples=100)
        feature_cols = _stage2_feature_columns()
        model = Stage2RF()
        model.train(df[feature_cols], df["danger_level"])

        shap_vals = model.shap_values(df[feature_cols].iloc[:3])
        assert "binary" in shap_vals
        sv = shap_vals["binary"]
        assert sv.shape == (3, len(feature_cols))
        prob_indices = [
            feature_cols.index("persistent_slab_prob"),
            feature_cols.index("storm_slab_prob"),
            feature_cols.index("loose_wet_prob"),
        ]
        for idx in prob_indices:
            assert sv[:, idx] is not None

    def test_three_class_output(self):
        from avalanche_ml.models.rf_stage2 import Stage2RF

        df = _make_synthetic_stage2_data(n_samples=200)
        feature_cols = _stage2_feature_columns()
        model = Stage2RF()
        model.train(df[feature_cols], df["danger_level"])

        result = model.predict(df[feature_cols].iloc[:10])
        assert "danger_3class" in result
        classes = result["danger_3class"]
        assert len(classes) == 10
        assert all(c in (0, 1, 2) for c in classes)

    def test_serialization_roundtrip(self):
        from avalanche_ml.models.rf_stage2 import Stage2RF

        df = _make_synthetic_stage2_data(n_samples=100)
        feature_cols = _stage2_feature_columns()
        model = Stage2RF()
        model.train(df[feature_cols], df["danger_level"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stage2_model"
            model.save(path)
            loaded = Stage2RF.load(path)

            result_orig = model.predict(df[feature_cols].iloc[:5])
            result_loaded = loaded.predict(df[feature_cols].iloc[:5])
            np.testing.assert_array_equal(
                result_orig["danger_binary"], result_loaded["danger_binary"]
            )
            np.testing.assert_allclose(
                result_orig["danger_distribution"],
                result_loaded["danger_distribution"],
            )

    def test_per_elevation_band_prediction(self):
        from avalanche_ml.models.rf_stage2 import Stage2RF

        df = _make_synthetic_stage2_data(n_samples=200)
        feature_cols = _stage2_feature_columns()

        for band in ELEVATION_BANDS:
            band_df = df[df["elevation_band"] == band]
            model = Stage2RF()
            model.train(band_df[feature_cols], band_df["danger_level"])

            result = model.predict(band_df[feature_cols].iloc[:3])
            assert "danger_binary" in result
            assert len(result["danger_binary"]) == 3


class TestTwoStagePipeline:
    def test_pipeline_end_to_end(self):
        from avalanche_ml.models.rf_stage2 import TwoStageRFPipeline

        df = _make_synthetic_stage2_data(n_samples=200)
        feature_cols = _feature_columns()

        pipeline = TwoStageRFPipeline()
        pipeline.train(
            df[feature_cols],
            df[PROBLEM_TYPE_FLAGS],
            df["danger_level"],
        )

        result = pipeline.predict(df[feature_cols].iloc[:10])
        assert "danger_binary" in result
        assert "danger_proba" in result
        assert "danger_distribution" in result
        assert "problem_types" in result
        assert "shap_values" in result
        assert len(result["danger_binary"]) == 10

    def test_pipeline_serialization_roundtrip(self):
        from avalanche_ml.models.rf_stage2 import TwoStageRFPipeline

        df = _make_synthetic_stage2_data(n_samples=150)
        feature_cols = _feature_columns()

        pipeline = TwoStageRFPipeline()
        pipeline.train(
            df[feature_cols],
            df[PROBLEM_TYPE_FLAGS],
            df["danger_level"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline"
            pipeline.save(path)
            loaded = TwoStageRFPipeline.load(path)

            result_orig = pipeline.predict(df[feature_cols].iloc[:5])
            result_loaded = loaded.predict(df[feature_cols].iloc[:5])
            np.testing.assert_array_equal(
                result_orig["danger_binary"], result_loaded["danger_binary"]
            )

    def test_pipeline_problem_types_in_output(self):
        from avalanche_ml.models.rf_stage2 import TwoStageRFPipeline

        df = _make_synthetic_stage2_data(n_samples=200)
        feature_cols = _feature_columns()

        pipeline = TwoStageRFPipeline()
        pipeline.train(
            df[feature_cols],
            df[PROBLEM_TYPE_FLAGS],
            df["danger_level"],
        )

        result = pipeline.predict(df[feature_cols].iloc[:5])
        probs = result["problem_types"]
        assert isinstance(probs, pd.DataFrame)
        assert list(probs.columns) == PROBLEM_TYPE_FLAGS
        assert (probs >= 0.0).all().all()
        assert (probs <= 1.0).all().all()

    def test_pipeline_shap_includes_problem_type_probs(self):
        from avalanche_ml.models.rf_stage2 import TwoStageRFPipeline

        df = _make_synthetic_stage2_data(n_samples=150)
        feature_cols = _feature_columns()

        pipeline = TwoStageRFPipeline()
        pipeline.train(
            df[feature_cols],
            df[PROBLEM_TYPE_FLAGS],
            df["danger_level"],
        )

        result = pipeline.predict(df[feature_cols].iloc[:3])
        shap_vals = result["shap_values"]
        assert "binary" in shap_vals
        stage2_features = feature_cols + [
            "persistent_slab_prob", "storm_slab_prob", "loose_wet_prob",
        ]
        assert shap_vals["binary"].shape == (3, len(stage2_features))

    def test_feature_importance_includes_problem_probs(self):
        from avalanche_ml.models.rf_stage2 import TwoStageRFPipeline

        rng = np.random.RandomState(123)
        n = 300
        df = _make_synthetic_stage2_data(n_samples=n, rng_seed=123)
        feature_cols = _feature_columns()

        df["persistent_slab"] = 0
        df["storm_slab"] = 0
        df["loose_wet"] = 0
        elevated = (
            (df["temp_gradient_days"] > 15)
            & (df[feature_cols[0]] > 0)
        )
        df.loc[elevated, "persistent_slab"] = 1
        danger = np.ones(n, dtype=int)
        danger[elevated.values] = rng.choice([3, 4, 5], size=elevated.sum())
        df["danger_level"] = danger

        pipeline = TwoStageRFPipeline()
        pipeline.train(
            df[feature_cols],
            df[PROBLEM_TYPE_FLAGS],
            df["danger_level"],
        )

        importance = pipeline.stage2.global_feature_importance()
        top_20_names = [name for name, _ in importance[:20]]
        prob_features = {"persistent_slab_prob", "storm_slab_prob", "loose_wet_prob"}
        found = prob_features & set(top_20_names)
        assert len(found) > 0, (
            f"Expected at least one problem-type prob in top 20, got: {top_20_names}"
        )
