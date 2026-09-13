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
