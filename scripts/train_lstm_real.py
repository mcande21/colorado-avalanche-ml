"""Train hierarchical LSTM on real SNOTEL/CAIC data; compare to RF baseline.

All sequences are precomputed in memory during init — no per-item overhead.
"""
from __future__ import annotations

import datetime
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import mlflow
from avalanche_ml.features.alignment import (
    ELEVATION_BANDS,
    PROBLEM_TYPE_FLAGS,
    TRAIN_END,
    VAL_END,
)
from avalanche_ml.features.physics import PHYSICS_FEATURES
from avalanche_ml.features.weather import get_feature_columns
from avalanche_ml.models.evaluation import (
    compute_binary_metrics,
    false_negative_rate_elevated,
    high_danger_detection_rate,
    ordinal_accuracy,
    per_elevation_band_metrics,
)
from avalanche_ml.models.lstm_model import HierarchicalLSTM
from avalanche_ml.models.lstm_train import combined_loss, save_model

DB_PATH = "data/avalanche.duckdb"
MODEL_DIR = "models"
DEVICE = "cpu"

BRANCH1_DAYS = 7
BRANCH2_DAYS = 30
AGGREGATION_DAYS = 3
MAX_SEASON_STEPS = 60

RF_BASELINE = {
    "threshold": 0.30,
    "val_binary_f1": 0.404,
    "val_fnr": 0.094,
    "val_binary_precision": 0.297,
    "val_binary_recall": 0.630,
}


def get_feature_cols() -> list[str]:
    return get_feature_columns() + PHYSICS_FEATURES


def load_all_data(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    feature_cols = get_feature_cols()
    cols = ["station_id", "date", "elevation_band", "danger_level"] + \
           PROBLEM_TYPE_FLAGS + feature_cols
    cols_sql = ", ".join(cols)
    df = conn.execute(f"""
        SELECT {cols_sql}
        FROM training_matrix
        ORDER BY station_id, date, elevation_band
    """).fetchdf()
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


class PrecomputedLSTMDataset(Dataset):
    """All branch tensors precomputed — __getitem__ is pure indexing."""

    def __init__(
        self,
        split_df: pd.DataFrame,
        full_df: pd.DataFrame,
        feature_cols: list[str],
        means: np.ndarray,
        stds: np.ndarray,
        max_samples: int | None = None,
    ):
        self.feature_cols = feature_cols
        self.n_features = len(feature_cols)
        self.means = means
        self.stds = stds

        station_arrays = self._build_station_arrays(full_df)

        unique = split_df.groupby(["station_id", "date", "elevation_band"]).first().reset_index()
        if max_samples and len(unique) > max_samples:
            unique = unique.sample(n=max_samples, random_state=42).reset_index(drop=True)

        n = len(unique)
        self.branch1 = np.zeros((n, BRANCH1_DAYS, self.n_features), dtype=np.float32)
        self.branch2 = np.zeros((n, BRANCH2_DAYS, self.n_features), dtype=np.float32)
        self.branch3 = np.zeros((n, MAX_SEASON_STEPS, self.n_features), dtype=np.float32)
        self.branch3_lengths = np.zeros(n, dtype=np.int64)
        self.danger_levels = np.zeros(n, dtype=np.int64)
        self.danger_binary = np.zeros(n, dtype=np.int64)
        self.problem_flags = np.zeros((n, len(PROBLEM_TYPE_FLAGS)), dtype=np.int64)
        self.station_ids = []
        self.dates = []
        self.elevation_bands = []

        for i, (_, row) in enumerate(unique.iterrows()):
            sid = row["station_id"]
            date = row["date"]
            if isinstance(date, pd.Timestamp):
                date = date.date()
            eb = row["elevation_band"]

            self.station_ids.append(sid)
            self.dates.append(str(date))
            self.elevation_bands.append(eb)

            danger = int(row["danger_level"])
            self.danger_levels[i] = danger
            self.danger_binary[i] = 1 if danger >= 3 else 0
            for j, pf in enumerate(PROBLEM_TYPE_FLAGS):
                self.problem_flags[i, j] = int(row[pf])

            if sid not in station_arrays:
                continue
            dates_arr, feats_arr = station_arrays[sid]

            self.branch1[i] = self._extract_branch(
                dates_arr, feats_arr, date, BRANCH1_DAYS,
            )
            self.branch2[i] = self._extract_branch(
                dates_arr, feats_arr, date, BRANCH2_DAYS,
            )
            b3, b3_len = self._extract_branch3(dates_arr, feats_arr, date)
            self.branch3[i] = b3
            self.branch3_lengths[i] = b3_len

        self.branch1 = torch.from_numpy(self.branch1)
        self.branch2 = torch.from_numpy(self.branch2)
        self.branch3 = torch.from_numpy(self.branch3)
        self.branch3_lengths = torch.from_numpy(self.branch3_lengths)

    def _build_station_arrays(self, df: pd.DataFrame) -> dict:
        """Build {station_id: (dates_array, features_array)} for fast slicing."""
        result = {}
        band0 = df[df["elevation_band"] == ELEVATION_BANDS[0]].copy()
        for sid, group in band0.groupby("station_id"):
            group = group.sort_values("date")
            dates = group["date"].values
            date_arr = np.array([
                pd.Timestamp(d).to_pydatetime().date()
                for d in dates
            ])
            feats = group[self.feature_cols].values.astype(np.float32)
            result[sid] = (date_arr, feats)
        return result

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        return np.nan_to_num((arr - self.means) / self.stds, nan=0.0)

    def _extract_branch(
        self, dates: np.ndarray, feats: np.ndarray, date, target_len: int,
    ) -> np.ndarray:
        start = date - datetime.timedelta(days=target_len - 1)
        mask = (dates >= start) & (dates <= date)
        subset = feats[mask]
        n = len(subset)
        if n >= target_len:
            arr = subset[-target_len:]
        elif n > 0:
            pad = np.zeros((target_len - n, self.n_features), dtype=np.float32)
            arr = np.vstack([pad, subset])
        else:
            arr = np.zeros((target_len, self.n_features), dtype=np.float32)
        return self._normalize(arr)

    def _extract_branch3(
        self, dates: np.ndarray, feats: np.ndarray, date,
    ) -> tuple[np.ndarray, int]:
        if date.month >= 10:
            season_start = datetime.date(date.year, 10, 1)
        else:
            season_start = datetime.date(date.year - 1, 10, 1)

        mask = (dates >= season_start) & (dates <= date)
        subset = feats[mask]

        if len(subset) == 0:
            return np.zeros((MAX_SEASON_STEPS, self.n_features), dtype=np.float32), 0

        n_days = len(subset)
        n_groups = (n_days + AGGREGATION_DAYS - 1) // AGGREGATION_DAYS
        agg = np.zeros((n_groups, self.n_features), dtype=np.float32)
        for k in range(n_groups):
            s = k * AGGREGATION_DAYS
            e = min(s + AGGREGATION_DAYS, n_days)
            with np.errstate(all="ignore"):
                agg[k] = np.nanmean(subset[s:e], axis=0)

        actual_len = n_groups
        agg = self._normalize(agg)

        result = np.zeros((MAX_SEASON_STEPS, self.n_features), dtype=np.float32)
        fill = min(actual_len, MAX_SEASON_STEPS)
        result[:fill] = agg[:fill]
        return result, min(actual_len, MAX_SEASON_STEPS)

    def __len__(self) -> int:
        return len(self.station_ids)

    def __getitem__(self, idx: int) -> dict:
        labels = {
            "danger_level": int(self.danger_levels[idx]),
            "danger_binary": int(self.danger_binary[idx]),
        }
        for j, pf in enumerate(PROBLEM_TYPE_FLAGS):
            labels[pf] = int(self.problem_flags[idx, j])

        return {
            "branch1": self.branch1[idx],
            "branch2": self.branch2[idx],
            "branch3": self.branch3[idx],
            "branch3_lengths": int(self.branch3_lengths[idx]),
            "labels": labels,
            "metadata": {
                "station_id": self.station_ids[idx],
                "date": self.dates[idx],
                "elevation_band": self.elevation_bands[idx],
            },
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "branch1": torch.stack([s["branch1"] for s in batch]),
        "branch2": torch.stack([s["branch2"] for s in batch]),
        "branch3": torch.stack([s["branch3"] for s in batch]),
        "branch3_lengths": torch.tensor(
            [s["branch3_lengths"] for s in batch], dtype=torch.long,
        ),
        "labels": {
            k: torch.tensor([s["labels"][k] for s in batch], dtype=torch.long)
            for k in batch[0]["labels"]
        },
        "metadata": {
            "station_id": [s["metadata"]["station_id"] for s in batch],
            "date": [s["metadata"]["date"] for s in batch],
            "elevation_band": [s["metadata"]["elevation_band"] for s in batch],
        },
    }


def train_loop(
    model: HierarchicalLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 10,
    device: str = "cpu",
) -> tuple[HierarchicalLSTM, dict]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            inputs = {
                "branch1": batch["branch1"].to(device),
                "branch2": batch["branch2"].to(device),
                "branch3": batch["branch3"].to(device),
                "branch3_lengths": batch["branch3_lengths"],
            }
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = combined_loss(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train)

        model.eval()
        val_total = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    "branch1": batch["branch1"].to(device),
                    "branch2": batch["branch2"].to(device),
                    "branch3": batch["branch3"].to(device),
                    "branch3_lengths": batch["branch3_lengths"],
                }
                labels = {k: v.to(device) for k, v in batch["labels"].items()}
                outputs = model(**inputs)
                loss = combined_loss(outputs, labels)
                val_total += loss.item()
                val_n += 1
        val_loss = val_total / max(val_n, 1)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            marker = " *"
        else:
            wait += 1

        print(f"  Epoch {epoch + 1:3d}: train={avg_train:.4f} val={val_loss:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e} ({elapsed:.1f}s){marker}")
        sys.stdout.flush()

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_all(
    model: HierarchicalLSTM, loader: DataLoader, device: str = "cpu",
) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            inputs = {
                "branch1": batch["branch1"].to(device),
                "branch2": batch["branch2"].to(device),
                "branch3": batch["branch3"].to(device),
                "branch3_lengths": batch["branch3_lengths"],
            }
            outputs = model(**inputs)
            bs = outputs["problem_types"].shape[0]
            for i in range(bs):
                row = {
                    "station_id": batch["metadata"]["station_id"][i],
                    "date": batch["metadata"]["date"][i],
                    "elevation_band": batch["metadata"]["elevation_band"][i],
                    "true_danger": batch["labels"]["danger_level"][i].item(),
                }
                row["prob_danger_binary"] = outputs["danger_binary"][i, 0].item()
                for j in range(outputs["danger_distribution"].shape[1]):
                    row[f"prob_danger_{j + 1}"] = outputs["danger_distribution"][i, j].item()
                for j, name in enumerate(PROBLEM_TYPE_FLAGS):
                    row[f"prob_{name}"] = outputs["problem_types"][i, j].item()
                rows.append(row)
    return pd.DataFrame(rows)


def evaluate_at_thresholds(preds_df: pd.DataFrame) -> dict:
    y_true_danger = preds_df["true_danger"].values
    y_true_binary = (y_true_danger >= 3).astype(int)
    prob_binary = preds_df["prob_danger_binary"].values

    results = {}
    for threshold in [0.25, 0.30, 0.35, 0.40, 0.50]:
        y_pred_binary = (prob_binary >= threshold).astype(int)
        bm = compute_binary_metrics(y_true_binary, y_pred_binary)
        fnr = false_negative_rate_elevated(y_true_danger, y_pred_binary)
        hdr = high_danger_detection_rate(y_true_danger, y_pred_binary)

        danger_cols = [f"prob_danger_{i}" for i in range(1, 6)]
        y_pred_danger = np.argmax(preds_df[danger_cols].values, axis=1) + 1
        ord_acc = ordinal_accuracy(y_true_danger, y_pred_danger)

        elev_metrics = per_elevation_band_metrics(
            y_true_danger, y_pred_danger, preds_df["elevation_band"].values,
        )

        results[threshold] = {
            "binary_f1": bm["f1"],
            "binary_precision": bm["precision"],
            "binary_recall": bm["recall"],
            "fnr": fnr,
            "high_danger_detection": hdr,
            "ordinal_accuracy": ord_acc,
            "per_elevation": elev_metrics,
        }
    return results


def print_comparison(val_results: dict, test_results: dict | None = None):
    print("\n" + "=" * 80)
    print("RF vs LSTM HEAD-TO-HEAD (Validation Set)")
    print("=" * 80)

    thresholds = sorted(val_results.keys())
    header = f"{'Metric':<25} {'RF (t=0.30)':<14}"
    for t in thresholds:
        header += f" {'LSTM t=' + f'{t:.2f}':<14}"
    print(header)
    print("-" * len(header))

    metrics = [
        ("Binary F1", "binary_f1", RF_BASELINE["val_binary_f1"]),
        ("Precision", "binary_precision", RF_BASELINE["val_binary_precision"]),
        ("Recall", "binary_recall", RF_BASELINE["val_binary_recall"]),
        ("FNR (lower=better)", "fnr", RF_BASELINE["val_fnr"]),
        ("Ordinal Accuracy", "ordinal_accuracy", None),
        ("High Danger Det.", "high_danger_detection", None),
    ]

    for name, key, rf_val in metrics:
        line = f"{name:<25}"
        if rf_val is not None:
            line += f" {rf_val:<14.3f}"
        else:
            line += f" {'N/A':<14}"
        for t in thresholds:
            v = val_results[t].get(key, 0)
            line += f" {v:<14.3f}"
        print(line)

    best_t = max(val_results.keys(), key=lambda t: val_results[t]["binary_f1"])
    best_f1 = val_results[best_t]["binary_f1"]
    rf_f1 = RF_BASELINE["val_binary_f1"]
    print(f"\nBest LSTM threshold: {best_t:.2f} (F1={best_f1:.3f})")
    print(f"RF baseline: t=0.30 (F1={rf_f1:.3f})")
    if best_f1 > rf_f1:
        print(f">>> LSTM wins by {best_f1 - rf_f1:.3f} F1 points")
    else:
        print(f">>> RF wins by {rf_f1 - best_f1:.3f} F1 points")

    if test_results:
        print("\n" + "=" * 80)
        print("TEST SET RESULTS (LSTM)")
        print("=" * 80)
        for t in sorted(test_results.keys()):
            r = test_results[t]
            print(f"  t={t:.2f}: F1={r['binary_f1']:.3f} Prec={r['binary_precision']:.3f} "
                  f"Rec={r['binary_recall']:.3f} FNR={r['fnr']:.3f} "
                  f"OrdAcc={r['ordinal_accuracy']:.3f}")

    print()
    if val_results:
        t30 = val_results.get(0.30, {})
        if t30:
            print("Per-elevation breakdown (LSTM t=0.30, val set):")
            for band, m in t30.get("per_elevation", {}).items():
                print(f"  {band}: OrdAcc={m['ordinal_accuracy']:.3f} "
                      f"MacroF1={m['macro_f1']:.3f} N={m['n_samples']}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    conn = duckdb.connect(DB_PATH, read_only=True)

    print("=" * 80)
    print("HIERARCHICAL LSTM TRAINING ON REAL DATA")
    print("=" * 80)
    sys.stdout.flush()

    feature_cols = get_feature_cols()
    n_features = len(feature_cols)
    print(f"Features: {n_features}")

    print("\nLoading all data into memory...")
    t0 = time.time()
    full_df = load_all_data(conn)
    print(f"  Loaded {len(full_df)} rows in {time.time() - t0:.1f}s")
    sys.stdout.flush()

    train_df = full_df[full_df["date"] <= pd.Timestamp(TRAIN_END)]
    val_df = full_df[(full_df["date"] > pd.Timestamp(TRAIN_END)) &
                     (full_df["date"] <= pd.Timestamp(VAL_END))]
    test_df = full_df[full_df["date"] > pd.Timestamp(VAL_END)]
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("\nComputing normalization stats...")
    train_vals = train_df[feature_cols].values.astype(np.float32)
    means = np.nan_to_num(np.nanmean(train_vals, axis=0), nan=0.0).astype(np.float32)
    stds = np.nan_to_num(np.nanstd(train_vals, axis=0), nan=1.0).astype(np.float32)
    stds[stds == 0] = 1.0

    max_train = 20_000
    max_val = 8_000

    print(f"\nPrecomputing sequences (max_train={max_train}, max_val={max_val})...")
    sys.stdout.flush()

    t0 = time.time()
    train_ds = PrecomputedLSTMDataset(train_df, full_df, feature_cols, means, stds, max_samples=max_train)
    print(f"  Train: {len(train_ds)} samples ({time.time() - t0:.1f}s)")
    sys.stdout.flush()

    t0 = time.time()
    val_ds = PrecomputedLSTMDataset(val_df, full_df, feature_cols, means, stds, max_samples=max_val)
    print(f"  Val: {len(val_ds)} samples ({time.time() - t0:.1f}s)")
    sys.stdout.flush()

    t0 = time.time()
    test_ds = PrecomputedLSTMDataset(test_df, full_df, feature_cols, means, stds)
    print(f"  Test: {len(test_ds)} samples ({time.time() - t0:.1f}s)")
    sys.stdout.flush()

    del full_df, train_df, val_df, test_df, train_vals

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"\nBatches: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    model = HierarchicalLSTM(input_size=n_features, hidden_size=64, num_layers=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} parameters")
    sys.stdout.flush()

    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}")
    sys.stdout.flush()

    mlflow.set_experiment("lstm-real-data")
    with mlflow.start_run(run_name="lstm-hierarchical-real"):
        mlflow.log_params({
            "model_type": "HierarchicalLSTM",
            "input_size": n_features,
            "hidden_size": 64,
            "num_layers": 2,
            "batch_size": batch_size,
            "device": DEVICE,
            "epochs_max": 50,
            "patience": 10,
            "lr": 1e-3,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
        })

        t0 = time.time()
        model, history = train_loop(
            model, train_loader, val_loader,
            epochs=50, lr=1e-3, patience=10, device=DEVICE,
        )
        train_time = time.time() - t0
        n_epochs = len(history["train_loss"])
        print(f"\nTraining completed: {n_epochs} epochs in {train_time:.0f}s")
        print(f"Best val loss: {min(history['val_loss']):.4f} "
              f"(epoch {np.argmin(history['val_loss']) + 1})")
        sys.stdout.flush()

        for i, (tl, vl) in enumerate(zip(history["train_loss"], history["val_loss"])):
            mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=i)

        save_model(model, f"{MODEL_DIR}/lstm_hierarchical_real.pt", metadata={
            "train_time_s": train_time,
            "epochs": n_epochs,
            "best_val_loss": min(history["val_loss"]),
            "train_samples": len(train_ds),
        })

        print(f"\n{'='*80}")
        print("EVALUATION")
        print(f"{'='*80}")
        sys.stdout.flush()

        print("Predicting on validation set...")
        val_preds = predict_all(model, val_loader, device=DEVICE)
        val_results = evaluate_at_thresholds(val_preds)

        print("Predicting on test set...")
        test_preds = predict_all(model, test_loader, device=DEVICE)
        test_results = evaluate_at_thresholds(test_preds)

        for threshold, metrics in val_results.items():
            prefix = f"val_t{threshold:.2f}"
            mlflow.log_metrics({
                f"{prefix}_binary_f1": metrics["binary_f1"],
                f"{prefix}_precision": metrics["binary_precision"],
                f"{prefix}_recall": metrics["binary_recall"],
                f"{prefix}_fnr": metrics["fnr"],
                f"{prefix}_ordinal_acc": metrics["ordinal_accuracy"],
            })

        for threshold, metrics in test_results.items():
            prefix = f"test_t{threshold:.2f}"
            mlflow.log_metrics({
                f"{prefix}_binary_f1": metrics["binary_f1"],
                f"{prefix}_precision": metrics["binary_precision"],
                f"{prefix}_recall": metrics["binary_recall"],
                f"{prefix}_fnr": metrics["fnr"],
                f"{prefix}_ordinal_acc": metrics["ordinal_accuracy"],
            })

        mlflow.log_metric("train_time_s", train_time)

        print_comparison(val_results, test_results)

    print("\nModel saved to models/lstm_hierarchical_real.pt")
    print("Results logged to MLflow experiment: lstm-real-data")


if __name__ == "__main__":
    main()
