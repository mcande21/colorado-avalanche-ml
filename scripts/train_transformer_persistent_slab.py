"""Train Transformer models for Persistent Slab prediction.

Per-band binary classification with grid search over lookback windows and
learning rates. Matches Schwartzreich et al. architecture: 64-unit, 2-layer,
4-head attention Transformer.

Persistent slab labels only exist for seasons 2014-15 through 2020-21,
so we use a custom temporal split within that range:
  Train: up to 2019-06-30 (seasons 14-15 through 18-19)
  Val:   2019-10-01 to 2020-06-30 (season 19-20)
  Test:  2020-10-01 to 2021-06-30 (season 20-21)

Usage:
    python scripts/train_transformer_persistent_slab.py [--db PATH]
"""
from __future__ import annotations

import argparse
import datetime
import sys
import time
from dataclasses import dataclass

import duckdb
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from avalanche_ml.features.alignment import ELEVATION_BANDS
from avalanche_ml.features.physics import PHYSICS_FEATURES
from avalanche_ml.features.weather import get_feature_columns
from avalanche_ml.models.lstm_train import FocalLoss, get_device
from avalanche_ml.models.transformer_model import PersistentSlabTransformer

PS_TRAIN_END = datetime.date(2019, 6, 30)
PS_VAL_START = datetime.date(2019, 10, 1)
PS_VAL_END = datetime.date(2020, 6, 30)
PS_TEST_START = datetime.date(2020, 10, 1)
PS_TEST_END = datetime.date(2021, 6, 30)

FEATURE_COLS: list[str] = []
N_FEATURES: int = 0


def init_feature_cols(conn: duckdb.DuckDBPyConnection) -> None:
    global FEATURE_COLS, N_FEATURES
    existing = {r[0] for r in conn.execute("DESCRIBE training_matrix").fetchall()}
    weather = get_feature_columns()
    physics = [c for c in PHYSICS_FEATURES if c in existing]
    FEATURE_COLS = weather + physics
    N_FEATURES = len(FEATURE_COLS)


LOOKBACKS = [7, 14, 30]
LEARNING_RATES = [1e-3, 5e-4]
BATCH_SIZE = 512
MAX_EPOCHS = 100
PATIENCE = 10


@dataclass
class StationTensors:
    features: torch.Tensor
    labels: torch.Tensor
    n: int


def load_and_normalize(
    conn: duckdb.DuckDBPyConnection,
    band: str,
    train_cutoff: datetime.date,
) -> tuple[dict[str, StationTensors], list[np.ndarray]]:
    cols_sql = ", ".join(FEATURE_COLS)
    rows = conn.execute(f"""
        SELECT station_id, date, {cols_sql}, persistent_slab
        FROM training_matrix
        WHERE elevation_band = ?
          AND date <= ?
        ORDER BY station_id, date
    """, [band, PS_TEST_END]).fetchnumpy()

    station_ids = rows["station_id"]
    all_dates = rows["date"]
    label_arr = rows["persistent_slab"].astype(np.float32)
    feat_matrix = np.column_stack(
        [rows[c].astype(np.float32) for c in FEATURE_COLS]
    )
    feat_matrix = np.nan_to_num(feat_matrix, nan=0.0)

    cutoff_np = np.datetime64(train_cutoff)
    train_mask = all_dates <= cutoff_np
    if train_mask.any():
        means = np.nanmean(feat_matrix[train_mask], axis=0).astype(np.float32)
        stds = np.nanstd(feat_matrix[train_mask], axis=0).astype(np.float32)
        stds[stds == 0] = 1.0
    else:
        means = np.zeros(N_FEATURES, dtype=np.float32)
        stds = np.ones(N_FEATURES, dtype=np.float32)

    feat_matrix = (feat_matrix - means) / stds
    feat_matrix = np.nan_to_num(feat_matrix, nan=0.0)

    stations: dict[str, StationTensors] = {}
    date_arrays: list[np.ndarray] = []
    unique_stations = np.unique(station_ids)

    for sid in unique_stations:
        mask = station_ids == sid
        f = torch.from_numpy(feat_matrix[mask].astype(np.float32))
        l = torch.from_numpy(label_arr[mask])
        stations[sid] = StationTensors(features=f, labels=l, n=len(f))
        date_arrays.append(all_dates[mask])

    return stations, date_arrays


def build_index(
    stations: dict[str, StationTensors],
    date_arrays: list[np.ndarray],
    start_date: datetime.date,
    end_date: datetime.date,
) -> list[tuple[str, int]]:
    start_np = np.datetime64(start_date)
    end_np = np.datetime64(end_date)
    index = []
    sids = sorted(stations.keys())
    for i, sid in enumerate(sids):
        dates = date_arrays[i]
        in_range = (dates >= start_np) & (dates <= end_np)
        for pos in np.where(in_range)[0]:
            index.append((sid, int(pos)))
    return index


class PrebuiltDataset(Dataset):
    def __init__(
        self,
        stations: dict[str, StationTensors],
        index: list[tuple[str, int]],
        lookback: int,
    ):
        self.stations = stations
        self.index = index
        self.lookback = lookback

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sid, pos = self.index[idx]
        st = self.stations[sid]
        lb = self.lookback
        start = pos - lb + 1

        if start >= 0:
            seq = st.features[start:pos + 1]
        else:
            real = st.features[:pos + 1]
            pad = torch.zeros(-start, st.features.shape[1])
            seq = torch.cat([pad, real], dim=0)

        return seq, st.labels[pos].unsqueeze(0)


@dataclass
class ExperimentResult:
    band: str
    lookback: int
    lr: float
    best_epoch: int
    train_f1: float
    val_f1: float
    val_precision: float
    val_recall: float
    test_f1: float
    test_precision: float
    test_recall: float
    train_time: float


def evaluate(
    model: PersistentSlabTransformer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x).squeeze(-1)
            preds = (out >= 0.5).int().cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(y.squeeze(-1).int().tolist())
    if not all_targets:
        return 0.0, 0.0, 0.0
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    return f1, prec, rec


def train_one(
    band: str,
    lookback: int,
    lr: float,
    stations: dict[str, StationTensors],
    date_arrays: list[np.ndarray],
    device: torch.device,
) -> ExperimentResult:
    train_idx = build_index(
        stations, date_arrays, datetime.date(2000, 1, 1), PS_TRAIN_END,
    )
    val_idx = build_index(stations, date_arrays, PS_VAL_START, PS_VAL_END)
    test_idx = build_index(stations, date_arrays, PS_TEST_START, PS_TEST_END)

    train_ds = PrebuiltDataset(stations, train_idx, lookback)
    val_ds = PrebuiltDataset(stations, val_idx, lookback)
    test_ds = PrebuiltDataset(stations, test_idx, lookback)

    print(f"    Sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    sys.stdout.flush()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = PersistentSlabTransformer(
        input_size=N_FEATURES, d_model=64, nhead=4, num_layers=2, dropout=0.3,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )
    criterion = FocalLoss(gamma=2.0)

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    best_epoch = 0

    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.squeeze(-1).to(device)
            optimizer.zero_grad()
            out = model(x).squeeze(-1)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.squeeze(-1).to(device)
                out = model(x).squeeze(-1)
                loss = criterion(out, y)
                val_loss += loss.item()
                val_n += 1
        avg_val_loss = val_loss / max(val_n, 1)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            best_epoch = epoch + 1
        else:
            wait += 1
            if wait >= PATIENCE:
                break

        if (epoch + 1) % 5 == 0 or wait >= PATIENCE:
            print(f"      Epoch {epoch + 1}: train={avg_train_loss:.4f} "
                  f"val={avg_val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")
            sys.stdout.flush()

    train_time = time.time() - t0
    print(f"      Stopped at epoch {best_epoch}, time={train_time:.1f}s")
    sys.stdout.flush()

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    train_f1, _, _ = evaluate(model, train_loader, device)
    val_f1, val_prec, val_rec = evaluate(model, val_loader, device)
    test_f1, test_prec, test_rec = evaluate(model, test_loader, device)

    return ExperimentResult(
        band=band, lookback=lookback, lr=lr, best_epoch=best_epoch,
        train_f1=train_f1, val_f1=val_f1, val_precision=val_prec, val_recall=val_rec,
        test_f1=test_f1, test_precision=test_prec, test_recall=test_rec,
        train_time=train_time,
    )


def main(db_path: str = "data/avalanche.duckdb") -> None:
    device = get_device()
    conn = duckdb.connect(db_path, read_only=True)
    init_feature_cols(conn)

    print(f"Device: {device}")
    print(f"Features: {N_FEATURES}")
    n_exp = len(ELEVATION_BANDS) * len(LOOKBACKS) * len(LEARNING_RATES)
    print(f"Grid: {len(ELEVATION_BANDS)} bands x {len(LOOKBACKS)} lookbacks "
          f"x {len(LEARNING_RATES)} LRs = {n_exp} experiments")
    print("Splits: train<=2019-06-30, val=2019-10 to 2020-06, test=2020-10 to 2021-06")
    print()
    sys.stdout.flush()

    results: list[ExperimentResult] = []

    for band in ELEVATION_BANDS:
        print(f"=== {band.upper()} ===")
        print("  Loading and normalizing...")
        sys.stdout.flush()
        stations, date_arrays = load_and_normalize(conn, band, PS_TRAIN_END)
        print(f"  Stations: {len(stations)}")
        sys.stdout.flush()

        for lookback in LOOKBACKS:
            for lr in LEARNING_RATES:
                print(f"  Training: lookback={lookback}, lr={lr}")
                sys.stdout.flush()
                result = train_one(
                    band, lookback, lr, stations, date_arrays, device,
                )
                results.append(result)
                print(f"    Val  F1={result.val_f1:.3f} P={result.val_precision:.3f} "
                      f"R={result.val_recall:.3f}")
                print(f"    Test F1={result.test_f1:.3f} P={result.test_precision:.3f} "
                      f"R={result.test_recall:.3f}")
                print()
                sys.stdout.flush()

    conn.close()
    print_report(results)


def print_report(results: list[ExperimentResult]) -> None:
    schwartz = {
        "below_treeline": 0.746,
        "near_treeline": 0.656,
        "above_treeline": 0.762,
    }
    rf_f1 = {
        "above_treeline": 0.787,
        "near_treeline": 0.787,
        "below_treeline": 0.786,
    }

    print("\n" + "=" * 95)
    print("PERSISTENT SLAB: TRANSFORMER vs RF vs SCHWARTZREICH")
    print("=" * 95)
    header = (f"{'Band':<18} {'LB':>3} {'LR':>7} {'Epoch':>5} "
              f"{'Val_F1':>7} {'Test_F1':>7} {'Test_P':>7} {'Test_R':>7} "
              f"{'RF_F1':>7} {'Schwz':>7} {'Time':>7}")
    print(header)
    print("-" * len(header))

    best_per_band: dict[str, ExperimentResult] = {}
    for r in results:
        print(f"{r.band:<18} {r.lookback:>3} {r.lr:>7.0e} {r.best_epoch:>5} "
              f"{r.val_f1:>7.3f} {r.test_f1:>7.3f} {r.test_precision:>7.3f} "
              f"{r.test_recall:>7.3f} {rf_f1[r.band]:>7.3f} "
              f"{schwartz[r.band]:>7.3f} {r.train_time:>6.1f}s")
        if r.band not in best_per_band or r.val_f1 > best_per_band[r.band].val_f1:
            best_per_band[r.band] = r

    print("\n" + "=" * 70)
    print("BEST PER BAND (by validation F1)")
    print("=" * 70)
    for band in ELEVATION_BANDS:
        b = best_per_band.get(band)
        if b:
            rf = rf_f1[band]
            sz = schwartz[band]
            winner = "Transformer" if b.test_f1 > rf else "RF"
            print(f"  {band}:")
            print(f"    Best config: lookback={b.lookback}, lr={b.lr}")
            print(f"    Transformer Test F1: {b.test_f1:.3f} "
                  f"(P={b.test_precision:.3f} R={b.test_recall:.3f})")
            print(f"    RF Stage-1 F1:       {rf:.3f}")
            print(f"    Schwartzreich F1:    {sz:.3f}")
            print(f"    Winner: {winner}")
            print()

    total_time = sum(r.train_time for r in results)
    print(f"Total training time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    if results:
        print(f"Average per model:   {total_time / len(results):.1f}s")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/avalanche.duckdb")
    args = parser.parse_args()
    main(db_path=args.db)
