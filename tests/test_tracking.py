from __future__ import annotations


class TestMLflowTracking:
    def test_configure_tracking(self, tmp_path):
        from avalanche_ml.models.tracking import configure_tracking

        tracking_uri = configure_tracking(str(tmp_path / "mlruns"))
        assert "sqlite" in tracking_uri

    def test_log_training_run(self, tmp_path):
        import mlflow
        from avalanche_ml.models.tracking import configure_tracking, log_training_run

        configure_tracking(str(tmp_path / "mlruns"))

        params = {"n_estimators": 500, "max_depth": 10}
        metrics = {
            "stage1_persistent_slab_f1": 0.85,
            "stage2_binary_f1": 0.78,
            "stage2_ordinal_accuracy": 0.92,
        }
        run_id = log_training_run(
            experiment_name="snowpack-instability-rf",
            params=params,
            metrics=metrics,
            tags={"model_type": "rf", "stage": "two_stage"},
        )
        assert run_id is not None

        run = mlflow.get_run(run_id)
        assert run.data.params["n_estimators"] == "500"
        assert float(run.data.metrics["stage1_persistent_slab_f1"]) == 0.85
        assert run.data.tags["model_type"] == "rf"

    def test_log_artifacts(self, tmp_path):
        import mlflow
        from avalanche_ml.models.tracking import configure_tracking, log_training_run

        configure_tracking(str(tmp_path / "mlruns"))

        model_file = tmp_path / "model.joblib"
        model_file.write_text("fake model")

        run_id = log_training_run(
            experiment_name="snowpack-instability-rf",
            params={"n_estimators": 100},
            metrics={"f1": 0.5},
            artifacts=[str(model_file)],
        )

        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_names = [a.path for a in artifacts]
        assert "model.joblib" in artifact_names

    def test_experiment_creation(self, tmp_path):
        import mlflow
        from avalanche_ml.models.tracking import configure_tracking, log_training_run

        configure_tracking(str(tmp_path / "mlruns"))

        log_training_run(
            experiment_name="test-experiment-creation",
            params={},
            metrics={"acc": 0.9},
        )

        experiment = mlflow.get_experiment_by_name("test-experiment-creation")
        assert experiment is not None

    def test_flatten_metrics(self):
        from avalanche_ml.models.tracking import flatten_metrics

        nested = {
            "stage1": {"persistent_slab": {"f1": 0.85, "precision": 0.90}},
            "stage2": {"binary_f1": 0.78},
        }
        flat = flatten_metrics(nested)
        assert flat["stage1_persistent_slab_f1"] == 0.85
        assert flat["stage1_persistent_slab_precision"] == 0.90
        assert flat["stage2_binary_f1"] == 0.78
