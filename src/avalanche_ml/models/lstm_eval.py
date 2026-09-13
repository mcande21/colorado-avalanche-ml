"""LSTM evaluation pipeline matching RF evaluation format."""
from __future__ import annotations

import numpy as np
import pandas as pd

from avalanche_ml.features.alignment import PROBLEM_TYPE_FLAGS
from avalanche_ml.models.evaluation import (
    evaluate_stage1,
    evaluate_stage2,
    per_elevation_band_metrics,
)
from avalanche_ml.models.lstm_model import HierarchicalLSTM
from avalanche_ml.models.lstm_train import get_device, predict, train_lstm
from avalanche_ml.models.tracking import configure_tracking, flatten_metrics, log_training_run


def evaluate_lstm(
    model: HierarchicalLSTM,
    dataloader,
    device: str = get_device(),
) -> dict:
    predictions = predict(model, dataloader, device=device)

    all_labels: dict[str, list] = {
        "danger_level": [],
        "elevation_band": [],
    }
    for pt in PROBLEM_TYPE_FLAGS:
        all_labels[pt] = []

    for batch in dataloader:
        bs = batch["labels"]["danger_level"].shape[0]
        for i in range(bs):
            all_labels["danger_level"].append(batch["labels"]["danger_level"][i].item())
            all_labels["elevation_band"].append(batch["metadata"]["elevation_band"][i])
            for pt in PROBLEM_TYPE_FLAGS:
                all_labels[pt].append(batch["labels"][pt][i].item())

    y_true_danger = np.array(all_labels["danger_level"])
    elevation_bands = np.array(all_labels["elevation_band"])

    y_true_pt = pd.DataFrame({pt: all_labels[pt] for pt in PROBLEM_TYPE_FLAGS})
    y_pred_pt = pd.DataFrame({
        pt: (predictions[f"prob_{pt}"].values >= 0.5).astype(int)
        for pt in PROBLEM_TYPE_FLAGS
    })
    stage1_metrics = evaluate_stage1(y_true_pt, y_pred_pt)

    y_pred_binary = (predictions["prob_danger_binary"].values >= 0.5).astype(int)
    danger_probs = predictions[[f"prob_danger_{i}" for i in range(1, 6)]].values
    y_pred_danger = danger_probs.argmax(axis=1) + 1
    y_pred_proba = predictions["prob_danger_binary"].values

    stage2_metrics = evaluate_stage2(
        y_true_danger, y_pred_binary, y_pred_danger, y_pred_proba,
    )

    band_metrics = per_elevation_band_metrics(
        y_true_danger, y_pred_danger, elevation_bands,
    )

    return {
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "per_elevation_band": band_metrics,
    }


def run_lstm_training_pipeline(
    train_loader,
    val_loader,
    test_loader,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    mlflow_tracking_dir: str | None = None,
    device: str = get_device(),
) -> dict:
    n_features = next(iter(train_loader))["branch1"].shape[2]
    model = HierarchicalLSTM(input_size=n_features)

    model, history = train_lstm(
        model, train_loader, val_loader,
        epochs=epochs, lr=lr, patience=patience, device=device,
    )

    train_metrics = evaluate_lstm(model, train_loader, device=device)
    val_metrics = evaluate_lstm(model, val_loader, device=device)
    test_metrics = evaluate_lstm(model, test_loader, device=device)

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
                if stage_name == "per_elevation_band":
                    for band_name, band_data in stage_metrics.items():
                        prefix = f"{split_name}_per_band_{band_name}"
                        flattened = flatten_metrics(band_data, prefix)
                        for k, v in flattened.items():
                            if isinstance(v, (int, float)) and not isinstance(v, bool):
                                flat_metrics[k] = v
                else:
                    prefix = f"{split_name}_{stage_name}"
                    flattened = flatten_metrics(stage_metrics, prefix)
                    for k, v in flattened.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            flat_metrics[k] = v

        params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience,
            "device": device,
        }
        tags = {"model_type": "lstm", "stage": "hierarchical", "split": "temporal"}

        mlflow_run_id = log_training_run(
            experiment_name="snowpack-instability-lstm",
            params=params,
            metrics=flat_metrics,
            tags=tags,
        )

    return {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "mlflow_run_id": mlflow_run_id,
    }
