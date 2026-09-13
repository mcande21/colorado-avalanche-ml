"""Per-band experiment battery targeting Schwartzreich 2026 benchmarks.

Groups A-G: hyperparameter grid, ExtraTrees, XGBoost, class weight ablation,
ensemble size, threshold optimization, physics feature update.
"""

import json
import os
import time
import warnings
from collections import defaultdict

import duckdb
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

import mlflow
from avalanche_ml.features.alignment import ELEVATION_BANDS, PROBLEM_TYPE_FLAGS
from avalanche_ml.features.physics import PHYSICS_FEATURES, compute_physics_features
from avalanche_ml.features.weather import get_feature_columns
from avalanche_ml.models.per_band_pipeline import (
    ENSEMBLE_CONFIGS,
    PROB_COLUMNS,
    STAGE1_DEFAULTS,
    _ensemble_predict_proba,
    _train_stage1_cell,
    staged_chronological_split,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

PROJECT_ROOT = "/Users/cooperanderson/work/personal/code/colorado-avalanche-ml"
SCHWARTZREICH = {"above_treeline": 0.508, "near_treeline": 0.525, "below_treeline": 0.544}
BAND_ABBREV = {"above_treeline": "ATL", "near_treeline": "NTL", "below_treeline": "BTL"}
DANGER_CLASSES = [1, 2, 3, 4]
RANDOM_STATE = 42
CHECKPOINT_FILE = f"{PROJECT_ROOT}/scripts/experiment_checkpoint.json"


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"bands": {}, "group_g": None}


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def strip_models(results):
    """Drop non-serializable sklearn/xgboost model objects before checkpointing."""
    return [{k: v for k, v in r.items() if k != "model"} for r in results]


def load_data():
    db = duckdb.connect(f"{PROJECT_ROOT}/data/avalanche.duckdb", read_only=True)
    df = db.execute("SELECT * FROM training_matrix").fetchdf()
    db.close()

    weather_cols = get_feature_columns()
    physics_cols = list(PHYSICS_FEATURES)
    feature_cols = [c for c in weather_cols + physics_cols if c in df.columns]
    non_null_cols = [c for c in feature_cols if df[c].notna().any()]
    print(f"Features: {len(non_null_cols)} usable / {len(feature_cols)} total")
    return df, non_null_cols


def evaluate_multiclass(y_true, y_pred):
    y_true = np.clip(y_true, 1, 4)
    y_pred = np.clip(y_pred, 1, 4)

    present = sorted(set(y_true) | set(y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))

    binary_f1 = float(f1_score(
        (y_true >= 3).astype(int), (y_pred >= 3).astype(int), zero_division=0
    ))

    fn_elevated = int(((y_true >= 3) & (y_pred < 3)).sum())
    total_elevated = int((y_true >= 3).sum())
    fnr = fn_elevated / total_elevated if total_elevated > 0 else 0.0

    per_class = {}
    for cls in DANGER_CLASSES:
        yt_bin = (y_true == cls).astype(int)
        yp_bin = (y_pred == cls).astype(int)
        if yt_bin.sum() == 0 and yp_bin.sum() == 0:
            per_class[cls] = 0.0
        else:
            per_class[cls] = float(f1_score(yt_bin, yp_bin, zero_division=0))

    return {
        "macro_f1": macro_f1,
        "binary_f1": binary_f1,
        "fnr": fnr,
        "per_class_f1": per_class,
        "class_3_f1": per_class.get(3, 0.0),
        "class_4_f1": per_class.get(4, 0.0),
    }


def prepare_band_data(df, band, feat_cols):
    band_df = df[df["elevation_band"] == band].copy()
    splits = staged_chronological_split(band_df)

    X_s1_train = splits["s1_train"][feat_cols].values.astype(np.float64)
    X_s1_eval = splits["s1_eval"][feat_cols].values.astype(np.float64)
    np.nan_to_num(X_s1_train, copy=False, nan=0.0)
    np.nan_to_num(X_s1_eval, copy=False, nan=0.0)

    ensembles = {}
    for pt in PROBLEM_TYPE_FLAGS:
        y_train = splits["s1_train"][pt].values.astype(int)
        y_eval = splits["s1_eval"][pt].values.astype(int)
        cell = _train_stage1_cell(
            X_s1_train, y_train, X_s1_eval, y_eval,
            ENSEMBLE_CONFIGS, STAGE1_DEFAULTS,
        )
        ensembles[pt] = cell["models"]

    def build_s2_features(split_key):
        X_raw = splits[split_key][feat_cols].values.astype(np.float64)
        np.nan_to_num(X_raw, copy=False, nan=0.0)
        s1_probs = {}
        for pt in PROBLEM_TYPE_FLAGS:
            s1_probs[f"{pt}_prob"] = _ensemble_predict_proba(ensembles[pt], X_raw)
        s1_arr = np.column_stack([s1_probs[c] for c in PROB_COLUMNS])
        X_full = np.hstack([X_raw, s1_arr])
        y = np.clip(splits[split_key]["danger_level"].values.astype(int), 1, 4)
        return X_full, y

    X_train, y_train = build_s2_features("s2_train")
    X_val, y_val = build_s2_features("s2_val")
    X_test, y_test = build_s2_features("s2_test")

    class_counts = np.bincount(y_train, minlength=5)
    total = len(y_train)
    scale_pos_weights = {}
    for cls in DANGER_CLASSES:
        n_cls = class_counts[cls]
        n_rest = total - n_cls
        scale_pos_weights[cls] = n_rest / n_cls if n_cls > 0 else 1.0

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "class_counts": class_counts,
        "scale_pos_weights": scale_pos_weights,
        "splits": splits,
        "ensembles": ensembles,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
    }


def train_and_eval(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    clf.fit(X_train, y_train)
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)
    val_metrics = evaluate_multiclass(y_val, val_pred)
    test_metrics = evaluate_multiclass(y_test, test_pred)
    return val_metrics, test_metrics, clf


def run_group_a(band_data, band):
    """Hyperparameter grid search — RF with Schwartzreich's grid."""
    n_estimators_grid = [200, 300, 500, 600]
    max_depth_grid = [6, 8, 10, 20, None]
    results = []

    for n_est in n_estimators_grid:
        for md in max_depth_grid:
            clf = RandomForestClassifier(
                n_estimators=n_est, max_depth=md,
                min_samples_leaf=20, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            )
            val_m, test_m, model = train_and_eval(
                clf, band_data["X_train"], band_data["y_train"],
                band_data["X_val"], band_data["y_val"],
                band_data["X_test"], band_data["y_test"],
            )
            config_str = f"RF(n={n_est},d={md})"
            results.append({
                "group": "A", "model_type": "RF", "config": config_str,
                "n_estimators": n_est, "max_depth": md,
                "val": val_m, "test": test_m, "model": model,
            })
    return results


def run_group_b(band_data, band):
    """ExtraTrees grid search."""
    n_estimators_grid = [200, 300, 500, 600]
    max_depth_grid = [6, 8, 10, 20, None]
    results = []

    for n_est in n_estimators_grid:
        for md in max_depth_grid:
            clf = ExtraTreesClassifier(
                n_estimators=n_est, max_depth=md,
                min_samples_leaf=20, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1,
            )
            val_m, test_m, model = train_and_eval(
                clf, band_data["X_train"], band_data["y_train"],
                band_data["X_val"], band_data["y_val"],
                band_data["X_test"], band_data["y_test"],
            )
            config_str = f"ET(n={n_est},d={md})"
            results.append({
                "group": "B", "model_type": "ExtraTrees", "config": config_str,
                "n_estimators": n_est, "max_depth": md,
                "val": val_m, "test": test_m, "model": model,
            })
    return results


def run_group_c(band_data, band):
    """XGBoost per-band."""
    n_estimators_grid = [300, 500]
    max_depth_grid = [6, 8, 10]
    lr_grid = [0.05, 0.1]
    results = []

    y_train = band_data["y_train"]
    class_counts = np.bincount(y_train, minlength=5)
    n_classes = len(DANGER_CLASSES)

    sample_weights = np.ones(len(y_train), dtype=np.float64)
    total = len(y_train)
    for cls in DANGER_CLASSES:
        n_cls = class_counts[cls]
        if n_cls > 0:
            w = total / (n_classes * n_cls)
            sample_weights[y_train == cls] = w

    for n_est in n_estimators_grid:
        for md in max_depth_grid:
            for lr in lr_grid:
                clf = XGBClassifier(
                    n_estimators=n_est, max_depth=md, learning_rate=lr,
                    objective="multi:softmax", num_class=5,
                    eval_metric="mlogloss",
                    early_stopping_rounds=20,
                    random_state=RANDOM_STATE, n_jobs=-1,
                    verbosity=0,
                )
                y_tr_shifted = y_train - 1
                y_val_shifted = band_data["y_val"] - 1
                clf.fit(
                    band_data["X_train"], y_tr_shifted,
                    eval_set=[(band_data["X_val"], y_val_shifted)],
                    sample_weight=sample_weights,
                    verbose=False,
                )
                val_pred = clf.predict(band_data["X_val"]) + 1
                test_pred = clf.predict(band_data["X_test"]) + 1
                val_m = evaluate_multiclass(band_data["y_val"], val_pred)
                test_m = evaluate_multiclass(band_data["y_test"], test_pred)

                config_str = f"XGB(n={n_est},d={md},lr={lr})"
                results.append({
                    "group": "C", "model_type": "XGBoost", "config": config_str,
                    "n_estimators": n_est, "max_depth": md, "learning_rate": lr,
                    "val": val_m, "test": test_m, "model": clf,
                })
    return results


def run_group_d(band_data, band, best_params):
    """Class weight ablation on best hyperparams from A-C."""
    n_est = best_params.get("n_estimators", 300)
    md = best_params.get("max_depth", 12)
    model_type = best_params.get("model_type", "RF")

    weight_configs = [
        ("none", None),
        ("balanced", "balanced"),
        ("1:3", {1: 1, 2: 1, 3: 3, 4: 3}),
        ("1:5", {1: 1, 2: 1, 3: 5, 4: 5}),
        ("1:10", {1: 1, 2: 1, 3: 10, 4: 10}),
    ]
    results = []

    for name, cw in weight_configs:
        if model_type == "ExtraTrees":
            clf = ExtraTreesClassifier(
                n_estimators=n_est, max_depth=md,
                min_samples_leaf=20, class_weight=cw,
                random_state=RANDOM_STATE, n_jobs=-1,
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=n_est, max_depth=md,
                min_samples_leaf=20, class_weight=cw,
                random_state=RANDOM_STATE, n_jobs=-1,
            )
        val_m, test_m, model = train_and_eval(
            clf, band_data["X_train"], band_data["y_train"],
            band_data["X_val"], band_data["y_val"],
            band_data["X_test"], band_data["y_test"],
        )
        config_str = f"{model_type}(cw={name})"
        results.append({
            "group": "D", "model_type": model_type, "config": config_str,
            "class_weight": name,
            "val": val_m, "test": test_m, "model": model,
        })
    return results


def run_group_e(band_data, band, feat_cols):
    """Ensemble size sweep for Stage-1 → Stage-2."""
    splits = band_data["splits"]
    X_s1_train = splits["s1_train"][feat_cols].values.astype(np.float64)
    X_s1_eval = splits["s1_eval"][feat_cols].values.astype(np.float64)
    np.nan_to_num(X_s1_train, copy=False, nan=0.0)
    np.nan_to_num(X_s1_eval, copy=False, nan=0.0)

    bigger_configs = [
        {"n_estimators": 100, "max_depth": 12},
        {"n_estimators": 200, "max_depth": 12},
        {"n_estimators": 300, "max_depth": 12},
        {"n_estimators": 400, "max_depth": 12},
        {"n_estimators": 500, "max_depth": 12},
        {"n_estimators": 600, "max_depth": 12},
        {"n_estimators": 700, "max_depth": 12},
    ]

    all_ensembles = {}
    for pt in PROBLEM_TYPE_FLAGS:
        y_train = splits["s1_train"][pt].values.astype(int)
        y_eval = splits["s1_eval"][pt].values.astype(int)
        cell = _train_stage1_cell(
            X_s1_train, y_train, X_s1_eval, y_eval,
            bigger_configs, STAGE1_DEFAULTS,
        )
        all_ensembles[pt] = cell["models"]

    def build_s2_with_topk(split_key, ensembles_topk):
        X_raw = splits[split_key][feat_cols].values.astype(np.float64)
        np.nan_to_num(X_raw, copy=False, nan=0.0)
        s1_probs = {}
        for pt in PROBLEM_TYPE_FLAGS:
            s1_probs[f"{pt}_prob"] = _ensemble_predict_proba(ensembles_topk[pt], X_raw)
        s1_arr = np.column_stack([s1_probs[c] for c in PROB_COLUMNS])
        X_full = np.hstack([X_raw, s1_arr])
        y = np.clip(splits[split_key]["danger_level"].values.astype(int), 1, 4)
        return X_full, y

    results = []
    for top_k in [1, 3, 5, 7]:
        topk_ensembles = {}
        for pt in PROBLEM_TYPE_FLAGS:
            n = min(top_k, len(all_ensembles[pt]))
            topk_ensembles[pt] = all_ensembles[pt][:n]

        X_train, y_train = build_s2_with_topk("s2_train", topk_ensembles)
        X_val, y_val = build_s2_with_topk("s2_val", topk_ensembles)
        X_test, y_test = build_s2_with_topk("s2_test", topk_ensembles)

        clf = RandomForestClassifier(
            n_estimators=300, max_depth=12,
            min_samples_leaf=20, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        val_m, test_m, model = train_and_eval(clf, X_train, y_train, X_val, y_val, X_test, y_test)
        config_str = f"Ensemble(top={top_k})"
        results.append({
            "group": "E", "model_type": "RF-Ensemble", "config": config_str,
            "top_k": top_k,
            "val": val_m, "test": test_m, "model": model,
        })
    return results


def run_group_f(band_data, best_model):
    """Threshold optimization per band."""
    thresholds = np.arange(0.20, 0.55, 0.05)
    results = []

    X_val = band_data["X_val"]
    y_val = band_data["y_val"]
    X_test = band_data["X_test"]
    y_test = band_data["y_test"]

    has_proba = hasattr(best_model, "predict_proba")
    if has_proba:
        val_proba = best_model.predict_proba(X_val)
        test_proba = best_model.predict_proba(X_test)
        classes = best_model.classes_
    else:
        return results

    for thresh in thresholds:
        val_pred = np.ones(len(y_val), dtype=int)
        test_pred = np.ones(len(y_test), dtype=int)

        for i, cls in enumerate(classes):
            if cls >= 3:
                val_mask = val_proba[:, i] >= thresh
                test_mask = test_proba[:, i] >= thresh
                for row_idx in range(len(val_pred)):
                    if val_mask[row_idx] and cls > val_pred[row_idx]:
                        val_pred[row_idx] = int(cls)
                for row_idx in range(len(test_pred)):
                    if test_mask[row_idx] and cls > test_pred[row_idx]:
                        test_pred[row_idx] = int(cls)

        val_pred_full = best_model.predict(X_val)
        test_pred_full = best_model.predict(X_test)
        for row_idx in range(len(val_pred)):
            base = int(val_pred_full[row_idx])
            if base < 3:
                val_pred[row_idx] = base
            else:
                elevated_proba = sum(
                    val_proba[row_idx, j] for j, c in enumerate(classes) if c >= 3
                )
                if elevated_proba >= thresh:
                    val_pred[row_idx] = base
                else:
                    val_pred[row_idx] = 2
        for row_idx in range(len(test_pred)):
            base = int(test_pred_full[row_idx])
            if base < 3:
                test_pred[row_idx] = base
            else:
                elevated_proba = sum(
                    test_proba[row_idx, j] for j, c in enumerate(classes) if c >= 3
                )
                if elevated_proba >= thresh:
                    test_pred[row_idx] = base
                else:
                    test_pred[row_idx] = 2

        val_m = evaluate_multiclass(y_val, val_pred)
        test_m = evaluate_multiclass(y_test, test_pred)
        config_str = f"Thresh({thresh:.2f})"
        results.append({
            "group": "F", "model_type": "Threshold", "config": config_str,
            "threshold": float(thresh),
            "val": val_m, "test": test_m,
        })
    return results


def run_group_g(df, feat_cols):
    """Recompute physics features and retrain best config."""
    print("\n--- Group G: Recomputing physics features ---")
    db = duckdb.connect(f"{PROJECT_ROOT}/data/avalanche.duckdb")
    n_rows = compute_physics_features(db)
    print(f"  Physics features recomputed: {n_rows} rows")
    db.close()

    df_new, new_feat_cols = load_data()
    results_by_band = {}

    for band in ELEVATION_BANDS:
        band_data = prepare_band_data(df_new, band, new_feat_cols)
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=12,
            min_samples_leaf=20, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        val_m, test_m, model = train_and_eval(
            clf, band_data["X_train"], band_data["y_train"],
            band_data["X_val"], band_data["y_val"],
            band_data["X_test"], band_data["y_test"],
        )
        results_by_band[band] = [{
            "group": "G", "model_type": "RF-PhysUpdate", "config": "RF(physics_update)",
            "val": val_m, "test": test_m, "model": model,
        }]
    return results_by_band


def print_summary_tables(all_results):
    print("\n" + "=" * 100)
    print("TABLE 1: BEST MACRO-F1 PER BAND (across all experiments)")
    print("=" * 100)
    header = f"{'Band':<6} {'Best_Config':<40} {'Val_MacF1':>10} {'Test_MacF1':>10} {'Schwartz':>10} {'Gap':>8}"
    print(header)
    print("-" * len(header))

    best_per_band = {}
    for band in ELEVATION_BANDS:
        abbrev = BAND_ABBREV[band]
        if band not in all_results:
            continue
        best = max(all_results[band], key=lambda r: r["val"]["macro_f1"])
        best_per_band[band] = best
        target = SCHWARTZREICH[band]
        gap = best["val"]["macro_f1"] - target
        print(
            f"{abbrev:<6} {best['config']:<40} "
            f"{best['val']['macro_f1']:>10.3f} {best['test']['macro_f1']:>10.3f} "
            f"{target:>10.3f} {gap:>+8.3f}"
        )

    print("\n" + "=" * 100)
    print("TABLE 2: PER-CLASS F1 FOR BEST CONFIG")
    print("=" * 100)
    header = f"{'Band':<6} {'Config':<35} {'Low(1)':>8} {'Mod(2)':>8} {'Cons(3)':>8} {'High(4)':>8}"
    print(header)
    print("-" * len(header))
    for band in ELEVATION_BANDS:
        abbrev = BAND_ABBREV[band]
        if band not in best_per_band:
            continue
        best = best_per_band[band]
        pc = best["val"]["per_class_f1"]
        print(
            f"{abbrev:<6} {best['config']:<35} "
            f"{pc.get(1, 0):>8.3f} {pc.get(2, 0):>8.3f} "
            f"{pc.get(3, 0):>8.3f} {pc.get(4, 0):>8.3f}"
        )

    print("\n" + "=" * 100)
    print("TABLE 3: PARETO FRONTIER (Macro-F1 vs FNR) — Val set")
    print("=" * 100)
    for band in ELEVATION_BANDS:
        abbrev = BAND_ABBREV[band]
        if band not in all_results:
            continue
        print(f"\n  {abbrev}:")
        band_results = sorted(all_results[band], key=lambda r: r["val"]["macro_f1"], reverse=True)

        pareto = []
        best_fnr = float("inf")
        for r in band_results:
            if r["val"]["fnr"] < best_fnr:
                pareto.append(r)
                best_fnr = r["val"]["fnr"]

        header = f"    {'Config':<40} {'MacF1':>8} {'FNR':>8} {'BinF1':>8}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for r in pareto[:10]:
            print(
                f"    {r['config']:<40} "
                f"{r['val']['macro_f1']:>8.3f} {r['val']['fnr']:>8.3f} "
                f"{r['val']['binary_f1']:>8.3f}"
            )

    print("\n" + "=" * 100)
    print("TABLE 4: WINNING MODEL TYPE PER BAND")
    print("=" * 100)
    header = f"{'Band':<6} {'Best RF':<30} {'Best ET':<30} {'Best XGB':<30} {'Winner':<10}"
    print(header)
    print("-" * len(header))
    for band in ELEVATION_BANDS:
        abbrev = BAND_ABBREV[band]
        if band not in all_results:
            continue
        type_best = {}
        for mtype in ["RF", "ExtraTrees", "XGBoost"]:
            typed = [r for r in all_results[band] if r["model_type"] == mtype]
            if typed:
                b = max(typed, key=lambda r: r["val"]["macro_f1"])
                type_best[mtype] = (b["config"], b["val"]["macro_f1"])
            else:
                type_best[mtype] = ("—", 0.0)

        winner = max(type_best, key=lambda k: type_best[k][1])
        print(
            f"{abbrev:<6} "
            f"{type_best['RF'][0]:<25}{type_best['RF'][1]:>5.3f} "
            f"{type_best['ExtraTrees'][0]:<25}{type_best['ExtraTrees'][1]:>5.3f} "
            f"{type_best['XGBoost'][0]:<25}{type_best['XGBoost'][1]:>5.3f} "
            f"{winner:<10}"
        )

    print("\n" + "=" * 100)
    print("TABLE 5: IMPACT OF CLASS WEIGHTING (Macro-F1 vs FNR)")
    print("=" * 100)
    for band in ELEVATION_BANDS:
        abbrev = BAND_ABBREV[band]
        if band not in all_results:
            continue
        d_results = [r for r in all_results[band] if r["group"] == "D"]
        if not d_results:
            continue
        print(f"\n  {abbrev}:")
        header = f"    {'Weight':<20} {'Val_MacF1':>10} {'Val_FNR':>10} {'Test_MacF1':>10} {'Test_FNR':>10}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for r in d_results:
            print(
                f"    {r.get('class_weight', '?'):<20} "
                f"{r['val']['macro_f1']:>10.3f} {r['val']['fnr']:>10.3f} "
                f"{r['test']['macro_f1']:>10.3f} {r['test']['fnr']:>10.3f}"
            )

    print("\n" + "=" * 100)
    print("TABLE 6: ALL RESULTS RANKED BY VAL MACRO-F1")
    print("=" * 100)
    for band in ELEVATION_BANDS:
        abbrev = BAND_ABBREV[band]
        if band not in all_results:
            continue
        ranked = sorted(all_results[band], key=lambda r: r["val"]["macro_f1"], reverse=True)
        print(f"\n  {abbrev} (top 15):")
        header = (
            f"    {'Grp':<4} {'Config':<40} "
            f"{'V_MacF1':>8} {'V_FNR':>7} {'V_C3':>6} {'V_C4':>6} "
            f"{'T_MacF1':>8} {'T_FNR':>7}"
        )
        print(header)
        print("    " + "-" * (len(header) - 4))
        for r in ranked[:15]:
            print(
                f"    {r['group']:<4} {r['config']:<40} "
                f"{r['val']['macro_f1']:>8.3f} {r['val']['fnr']:>7.3f} "
                f"{r['val']['class_3_f1']:>6.3f} {r['val']['class_4_f1']:>6.3f} "
                f"{r['test']['macro_f1']:>8.3f} {r['test']['fnr']:>7.3f}"
            )


def main():
    t0 = time.time()
    df, feat_cols = load_data()

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
    mlflow.set_experiment("gap-experiments-v1")

    checkpoint = load_checkpoint()
    all_results = defaultdict(list)

    for band in ELEVATION_BANDS:
        abbrev = BAND_ABBREV[band]

        if abbrev in checkpoint["bands"]:
            print(f"\nSkipping {abbrev} — already completed (checkpoint)")
            all_results[band] = checkpoint["bands"][abbrev]
            continue

        print(f"\n{'='*80}")
        print(f"BAND: {abbrev} ({band})")
        print(f"{'='*80}")

        band_data = prepare_band_data(df, band, feat_cols)
        print(f"  Train: {band_data['n_train']}, Val: {band_data['n_val']}, Test: {band_data['n_test']}")
        print(f"  Class dist (train): {dict(enumerate(band_data['class_counts']))}")

        # --- Group A: RF grid ---
        print("\n--- Group A: RF hyperparameter grid (20 configs) ---")
        t1 = time.time()
        a_results = run_group_a(band_data, band)
        all_results[band].extend(a_results)
        best_a = max(a_results, key=lambda r: r["val"]["macro_f1"])
        print(f"  Best RF: {best_a['config']} val_macro_f1={best_a['val']['macro_f1']:.3f} [{time.time()-t1:.0f}s]")

        # --- Group B: ExtraTrees grid ---
        print("\n--- Group B: ExtraTrees grid (20 configs) ---")
        t1 = time.time()
        b_results = run_group_b(band_data, band)
        all_results[band].extend(b_results)
        best_b = max(b_results, key=lambda r: r["val"]["macro_f1"])
        print(f"  Best ET: {best_b['config']} val_macro_f1={best_b['val']['macro_f1']:.3f} [{time.time()-t1:.0f}s]")

        # --- Group C: XGBoost ---
        print("\n--- Group C: XGBoost grid (12 configs) ---")
        t1 = time.time()
        c_results = run_group_c(band_data, band)
        all_results[band].extend(c_results)
        best_c = max(c_results, key=lambda r: r["val"]["macro_f1"])
        print(f"  Best XGB: {best_c['config']} val_macro_f1={best_c['val']['macro_f1']:.3f} [{time.time()-t1:.0f}s]")

        # --- Identify best from A-C for Group D ---
        all_abc = a_results + b_results + c_results
        best_abc = max(all_abc, key=lambda r: r["val"]["macro_f1"])
        best_params = {
            "n_estimators": best_abc.get("n_estimators", 300),
            "max_depth": best_abc.get("max_depth", 12),
            "model_type": best_abc["model_type"],
        }
        print(f"\n  Best A-C: {best_abc['config']} (macro_f1={best_abc['val']['macro_f1']:.3f})")

        # --- Group D: Class weight ablation ---
        print("\n--- Group D: Class weight ablation (5 configs) ---")
        t1 = time.time()
        d_results = run_group_d(band_data, band, best_params)
        all_results[band].extend(d_results)
        best_d = max(d_results, key=lambda r: r["val"]["macro_f1"])
        print(f"  Best CW: {best_d['config']} val_macro_f1={best_d['val']['macro_f1']:.3f} [{time.time()-t1:.0f}s]")

        # --- Group E: Ensemble size ---
        print("\n--- Group E: Ensemble size sweep ---")
        t1 = time.time()
        e_results = run_group_e(band_data, band, feat_cols)
        all_results[band].extend(e_results)
        best_e = max(e_results, key=lambda r: r["val"]["macro_f1"])
        print(f"  Best Ensemble: {best_e['config']} val_macro_f1={best_e['val']['macro_f1']:.3f} [{time.time()-t1:.0f}s]")

        # --- Group F: Threshold optimization ---
        print("\n--- Group F: Threshold optimization ---")
        t1 = time.time()
        overall_best = max(all_results[band], key=lambda r: r["val"]["macro_f1"])
        if hasattr(overall_best.get("model"), "predict_proba"):
            f_results = run_group_f(band_data, overall_best["model"])
            all_results[band].extend(f_results)
            if f_results:
                best_f = max(f_results, key=lambda r: r["val"]["macro_f1"])
                print(f"  Best Thresh: {best_f['config']} val_macro_f1={best_f['val']['macro_f1']:.3f} [{time.time()-t1:.0f}s]")
            else:
                print(f"  No threshold results [{time.time()-t1:.0f}s]")
        else:
            print("  Skipped (best model lacks predict_proba)")

        checkpoint["bands"][abbrev] = strip_models(all_results[band])
        save_checkpoint(checkpoint)

    # --- Group G: Physics features update ---
    if checkpoint.get("group_g") is not None:
        print("\nSkipping Group G — already completed (checkpoint)")
        g_results = checkpoint["group_g"]
    else:
        print(f"\n{'='*80}")
        print("Group G: Physics feature update + retrain")
        print(f"{'='*80}")
        t1 = time.time()
        g_results = run_group_g(df, feat_cols)
        checkpoint["group_g"] = {band: strip_models(res) for band, res in g_results.items()}
        save_checkpoint(checkpoint)
        print(f"  [{time.time()-t1:.0f}s]")

    for band, results in g_results.items():
        all_results[band].extend(results)
    for band in ELEVATION_BANDS:
        abbrev = BAND_ABBREV[band]
        if g_results.get(band):
            r = g_results[band][0]
            print(f"  {abbrev}: val_macro_f1={r['val']['macro_f1']:.3f}")

    # --- Log best per band to MLflow ---
    for band in ELEVATION_BANDS:
        if band not in all_results:
            continue
        best = max(all_results[band], key=lambda r: r["val"]["macro_f1"])
        with mlflow.start_run(run_name=f"best_{BAND_ABBREV[band]}"):
            mlflow.log_params({
                "band": band,
                "config": best["config"],
                "group": best["group"],
                "model_type": best["model_type"],
            })
            mlflow.log_metrics({
                "val_macro_f1": best["val"]["macro_f1"],
                "val_binary_f1": best["val"]["binary_f1"],
                "val_fnr": best["val"]["fnr"],
                "val_class3_f1": best["val"]["class_3_f1"],
                "val_class4_f1": best["val"]["class_4_f1"],
                "test_macro_f1": best["test"]["macro_f1"],
                "test_binary_f1": best["test"]["binary_f1"],
                "test_fnr": best["test"]["fnr"],
                "test_class3_f1": best["test"]["class_3_f1"],
                "test_class4_f1": best["test"]["class_4_f1"],
                "schwartzreich_target": SCHWARTZREICH[band],
                "gap": best["val"]["macro_f1"] - SCHWARTZREICH[band],
            })

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Total models trained: {sum(len(v) for v in all_results.values())}")

    print_summary_tables(all_results)


if __name__ == "__main__":
    main()
