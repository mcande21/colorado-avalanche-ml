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


def _make_per_band_data(
    n_per_band: int = 300,
    rng_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(rng_seed)
    weather_cols = get_feature_columns()
    physics_cols = list(PHYSICS_FEATURES)
    frames = []
    for band in ELEVATION_BANDS:
        data: dict[str, np.ndarray] = {}
        for col in weather_cols:
            data[col] = rng.randn(n_per_band).astype(np.float64)
        for col in physics_cols:
            data[col] = rng.randn(n_per_band).astype(np.float64)

        data["precip_sum_72h"] = rng.uniform(0, 5, n_per_band)
        data["temp_gradient_days"] = rng.uniform(0, 30, n_per_band)
        data["temp_mean_24h"] = rng.uniform(-10, 50, n_per_band)

        data["storm_slab"] = (data["precip_sum_72h"] > 2.5).astype(int)
        data["persistent_slab"] = (data["temp_gradient_days"] > 15).astype(int)
        data["loose_wet"] = (data["temp_mean_24h"] > 35).astype(int)

        danger = np.ones(n_per_band, dtype=int)
        danger[data["precip_sum_72h"] > 2.0] = 2
        danger[data["precip_sum_72h"] > 3.5] = 3
        danger[data["temp_gradient_days"] > 25] = 4
        data["danger_level"] = danger

        base = datetime.date(2015, 11, 1)
        data["date"] = pd.array(
            [base + datetime.timedelta(days=int(i)) for i in range(n_per_band)]
        )
        data["station_id"] = np.array(["STATION_01"] * n_per_band)
        data["elevation_band"] = np.array([band] * n_per_band)
        data["zone_id"] = np.array(["ZONE_01"] * n_per_band)

        frames.append(pd.DataFrame(data))

    return pd.concat(frames, ignore_index=True)


class TestPerBandFiltering:
    def test_filter_produces_correct_subset(self):
        df = _make_per_band_data(n_per_band=100)
        for band in ELEVATION_BANDS:
            subset = df[df["elevation_band"] == band]
            assert len(subset) == 100
            assert (subset["elevation_band"] == band).all()


class TestStagedChronologicalSplit:
    def test_split_dates_correct(self):
        from avalanche_ml.models.per_band_pipeline import staged_chronological_split

        df = _make_per_band_data(n_per_band=300)
        band_df = df[df["elevation_band"] == "above_treeline"].copy()
        splits = staged_chronological_split(band_df)

        assert len(splits["s1_train"]) > 0
        assert len(splits["s1_eval"]) > 0
        assert len(splits["s2_train"]) > 0
        assert len(splits["s2_val"]) > 0
        assert len(splits["s2_test"]) > 0

        s1_train_max = pd.to_datetime(splits["s1_train"]["date"]).max()
        s1_eval_min = pd.to_datetime(splits["s1_eval"]["date"]).min()
        assert s1_train_max < s1_eval_min

        s2_train_max = pd.to_datetime(splits["s2_train"]["date"]).max()
        s2_val_min = pd.to_datetime(splits["s2_val"]["date"]).min()
        assert s2_train_max < s2_val_min


class TestFrozenStage1Predictions:
    def test_stage2_never_sees_actual_labels(self):
        from avalanche_ml.models.per_band_pipeline import PerBandPipeline

        df = _make_per_band_data(n_per_band=300)
        pipeline = PerBandPipeline()
        results = pipeline.train(df)

        for band in ELEVATION_BANDS:
            band_result = results[band]
            s2_features = band_result["s2_feature_columns"]
            for pt in PROBLEM_TYPE_FLAGS:
                assert pt not in s2_features
            for pt in PROBLEM_TYPE_FLAGS:
                assert f"{pt}_prob" in s2_features


class TestEnsembleAveraging:
    def test_top3_configs_averaged(self):
        from avalanche_ml.models.per_band_pipeline import PerBandPipeline

        df = _make_per_band_data(n_per_band=300)
        pipeline = PerBandPipeline()
        results = pipeline.train(df)

        for band in ELEVATION_BANDS:
            for pt in PROBLEM_TYPE_FLAGS:
                ensemble_info = results[band]["stage1_ensembles"][pt]
                assert ensemble_info["n_configs_trained"] >= 1
                assert ensemble_info["n_configs_kept"] >= 1


class TestPipelineEndToEnd:
    def test_train_and_predict(self):
        from avalanche_ml.models.per_band_pipeline import PerBandPipeline

        df = _make_per_band_data(n_per_band=300)
        pipeline = PerBandPipeline()
        results = pipeline.train(df)

        assert len(results) == 3
        for band in ELEVATION_BANDS:
            assert band in results
            assert "stage2_val_metrics" in results[band]
            assert "stage2_test_metrics" in results[band]

        test_df = df[df["elevation_band"] == "above_treeline"].iloc[:10].copy()
        preds = pipeline.predict(test_df, "above_treeline")
        assert "danger_level" in preds
        assert "danger_proba" in preds
        assert "problem_type_probs" in preds
        assert len(preds["danger_level"]) == 10

    def test_danger_levels_in_valid_range(self):
        from avalanche_ml.models.per_band_pipeline import PerBandPipeline

        df = _make_per_band_data(n_per_band=300)
        pipeline = PerBandPipeline()
        pipeline.train(df)

        test_df = df[df["elevation_band"] == "near_treeline"].iloc[:20].copy()
        preds = pipeline.predict(test_df, "near_treeline")
        for dl in preds["danger_level"]:
            assert 1 <= dl <= 4


class TestSerialization:
    def test_save_and_load_roundtrip(self):
        from avalanche_ml.models.per_band_pipeline import PerBandPipeline

        df = _make_per_band_data(n_per_band=300)
        pipeline = PerBandPipeline()
        pipeline.train(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "pipeline"
            pipeline.save(save_path)

            loaded = PerBandPipeline.load(save_path)

            test_df = df[df["elevation_band"] == "below_treeline"].iloc[:10].copy()
            orig = pipeline.predict(test_df, "below_treeline")
            loaded_preds = loaded.predict(test_df, "below_treeline")

            np.testing.assert_array_equal(
                orig["danger_level"], loaded_preds["danger_level"]
            )


def _make_sequential_per_band_data(
    n_per_band: int = 300,
    n_stations: int = 3,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic data with multiple stations for sequence construction."""
    rng = np.random.RandomState(rng_seed)
    weather_cols = get_feature_columns()
    physics_cols = list(PHYSICS_FEATURES)
    frames = []
    days_per_station = n_per_band // n_stations

    for band in ELEVATION_BANDS:
        for s in range(n_stations):
            data: dict[str, np.ndarray] = {}
            n = days_per_station
            for col in weather_cols:
                data[col] = rng.randn(n).astype(np.float64)
            for col in physics_cols:
                data[col] = rng.randn(n).astype(np.float64)

            data["precip_sum_72h"] = rng.uniform(0, 5, n)
            data["temp_gradient_days"] = rng.uniform(0, 30, n)
            data["temp_mean_24h"] = rng.uniform(-10, 50, n)

            data["storm_slab"] = (data["precip_sum_72h"] > 2.5).astype(int)
            data["persistent_slab"] = (data["temp_gradient_days"] > 15).astype(int)
            data["loose_wet"] = (data["temp_mean_24h"] > 35).astype(int)

            danger = np.ones(n, dtype=int)
            danger[data["precip_sum_72h"] > 2.0] = 2
            danger[data["precip_sum_72h"] > 3.5] = 3
            danger[data["temp_gradient_days"] > 25] = 4
            data["danger_level"] = danger

            base = datetime.date(2015, 11, 1)
            data["date"] = pd.array(
                [base + datetime.timedelta(days=int(i)) for i in range(n)]
            )
            data["station_id"] = np.array([f"STATION_{s:02d}"] * n)
            data["elevation_band"] = np.array([band] * n)
            data["zone_id"] = np.array(["ZONE_01"] * n)
            frames.append(pd.DataFrame(data))

    return pd.concat(frames, ignore_index=True)


class TestHybridStage1UsesTransformer:
    def test_hybrid_stage1_uses_transformer_for_persistent_slab(self):
        from avalanche_ml.models.per_band_pipeline import PerBandPipeline

        df = _make_sequential_per_band_data(n_per_band=300, n_stations=3)
        pipeline = PerBandPipeline(persistent_slab_model="transformer")

        assert pipeline.persistent_slab_model == "transformer"

        pipeline.train(df)

        for band in ELEVATION_BANDS:
            assert band in pipeline.transformer_models_
            model = pipeline.transformer_models_[band]
            assert hasattr(model, "transformer")


class TestHybridStage1UsesRFForOthers:
    def test_hybrid_stage1_uses_rf_for_other_problem_types(self):
        from avalanche_ml.models.per_band_pipeline import PerBandPipeline

        df = _make_sequential_per_band_data(n_per_band=300, n_stations=3)
        pipeline = PerBandPipeline(persistent_slab_model="transformer")
        pipeline.train(df)

        for band in ELEVATION_BANDS:
            for pt in ["storm_slab", "loose_wet"]:
                assert pt in pipeline.stage1_ensembles_[band]
                models = pipeline.stage1_ensembles_[band][pt]
                assert len(models) >= 1
            assert "persistent_slab" not in pipeline.stage1_ensembles_[band]


class TestHybridStage1PredictionsShape:
    def test_hybrid_stage1_predictions_shape(self):
        from avalanche_ml.models.per_band_pipeline import PerBandPipeline

        df = _make_sequential_per_band_data(n_per_band=300, n_stations=3)
        pipeline = PerBandPipeline(persistent_slab_model="transformer")
        pipeline.train(df)

        for band in ELEVATION_BANDS:
            test_df = df[df["elevation_band"] == band].iloc[:10].copy()
            preds = pipeline.predict(test_df, band)
            for pt in PROBLEM_TYPE_FLAGS:
                assert pt in preds["problem_type_probs"]
                probs = preds["problem_type_probs"][pt]
                assert len(probs) == 10
                for p in probs:
                    assert 0.0 <= p <= 1.0


class TestHybridStage2ReceivesTransformerPredictions:
    def test_hybrid_stage2_receives_transformer_predictions(self):
        from avalanche_ml.models.per_band_pipeline import PerBandPipeline

        df = _make_sequential_per_band_data(n_per_band=300, n_stations=3)
        pipeline = PerBandPipeline(persistent_slab_model="transformer")
        results = pipeline.train(df)

        for band in ELEVATION_BANDS:
            s2_features = results[band]["s2_feature_columns"]
            assert "persistent_slab_prob" in s2_features
            assert "storm_slab_prob" in s2_features
            assert "loose_wet_prob" in s2_features
            for pt in PROBLEM_TYPE_FLAGS:
                assert pt not in s2_features


class TestHybridPipelineEndToEnd:
    def test_hybrid_pipeline_end_to_end(self):
        from avalanche_ml.models.per_band_pipeline import PerBandPipeline

        df = _make_sequential_per_band_data(n_per_band=300, n_stations=3)
        pipeline = PerBandPipeline(persistent_slab_model="transformer")
        results = pipeline.train(df)

        assert len(results) == 3
        for band in ELEVATION_BANDS:
            assert "stage2_val_metrics" in results[band]
            assert "stage2_test_metrics" in results[band]
            assert results[band]["stage2_test_metrics"]["macro_f1"] >= 0.0

            test_df = df[df["elevation_band"] == band].iloc[:10].copy()
            preds = pipeline.predict(test_df, band)
            assert len(preds["danger_level"]) == 10
            for dl in preds["danger_level"]:
                assert 1 <= dl <= 4


class TestHybridPipelineSerialization:
    def test_hybrid_pipeline_serialization(self):
        from avalanche_ml.models.per_band_pipeline import PerBandPipeline

        df = _make_sequential_per_band_data(n_per_band=300, n_stations=3)
        pipeline = PerBandPipeline(persistent_slab_model="transformer")
        pipeline.train(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "hybrid_pipeline"
            pipeline.save(save_path)

            loaded = PerBandPipeline.load(save_path)

            assert loaded.persistent_slab_model == "transformer"
            assert len(loaded.transformer_models_) == 3

            test_df = df[df["elevation_band"] == "above_treeline"].iloc[:10].copy()
            orig = pipeline.predict(test_df, "above_treeline")
            loaded_preds = loaded.predict(test_df, "above_treeline")

            np.testing.assert_array_equal(
                orig["danger_level"], loaded_preds["danger_level"]
            )


class TestGRUStage2PipelineEndToEnd:
    def test_pipeline_gru_stage2_trains_and_predicts(self):
        from avalanche_ml.models.per_band_pipeline import PerBandPipeline

        df = _make_sequential_per_band_data(n_per_band=300, n_stations=3)
        pipeline = PerBandPipeline(
            persistent_slab_model="transformer", stage2_model="gru",
        )
        results = pipeline.train(df)

        assert len(results) == 3
        for band in ELEVATION_BANDS:
            assert "stage2_val_metrics" in results[band]
            assert results[band]["stage2_val_metrics"]["macro_f1"] >= 0.0

            test_df = df[df["elevation_band"] == band].iloc[:10].copy()
            preds = pipeline.predict(test_df, band)
            assert len(preds["danger_level"]) == 10
            for dl in preds["danger_level"]:
                assert 1 <= dl <= 4
            assert preds["danger_proba"].shape == (10, 4)
            row_sums = preds["danger_proba"].sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


class TestGRUStage2Serialization:
    def test_gru_pipeline_save_load_roundtrip(self):
        from avalanche_ml.models.per_band_pipeline import PerBandPipeline

        df = _make_sequential_per_band_data(n_per_band=300, n_stations=3)
        pipeline = PerBandPipeline(
            persistent_slab_model="transformer", stage2_model="gru",
        )
        pipeline.train(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "gru_pipeline"
            pipeline.save(save_path)
            loaded = PerBandPipeline.load(save_path)

            assert loaded.stage2_model == "gru"

            test_df = df[df["elevation_band"] == "above_treeline"].iloc[:10].copy()
            orig = pipeline.predict(test_df, "above_treeline")
            loaded_preds = loaded.predict(test_df, "above_treeline")

            np.testing.assert_array_equal(
                orig["danger_level"], loaded_preds["danger_level"],
            )


class TestGRUReceivesS1ProbsInPipeline:
    def test_s2_features_include_s1_probs(self):
        from avalanche_ml.models.per_band_pipeline import PerBandPipeline

        df = _make_sequential_per_band_data(n_per_band=300, n_stations=3)
        pipeline = PerBandPipeline(
            persistent_slab_model="transformer", stage2_model="gru",
        )
        results = pipeline.train(df)

        for band in ELEVATION_BANDS:
            s2_features = results[band]["s2_feature_columns"]
            assert "persistent_slab_prob" in s2_features
            assert "storm_slab_prob" in s2_features
            assert "loose_wet_prob" in s2_features
