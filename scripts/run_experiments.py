"""RF hyperparameter experiment battery for avalanche danger prediction."""

import warnings

import duckdb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import mlflow
from avalanche_ml.features.physics import PHYSICS_FEATURES
from avalanche_ml.features.weather import get_feature_columns

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = "/Users/cooperanderson/work/personal/code/colorado-avalanche-ml"


def load_data():
    db = duckdb.connect(f"{PROJECT_ROOT}/data/avalanche.duckdb", read_only=True)
    df = db.execute("SELECT * FROM training_matrix").fetchdf()
    db.close()

    weather_cols = get_feature_columns()
    physics_cols = list(PHYSICS_FEATURES)
    feature_cols = [c for c in weather_cols + physics_cols if c in df.columns]
    non_null_cols = [c for c in feature_cols if df[c].notna().any()]
    print(f"Using {len(non_null_cols)} features (dropped {len(feature_cols) - len(non_null_cols)} all-null)")

    df["elevated"] = (df["danger_level"] >= 3).astype(int)

    train = df[df["date"] <= "2024-06-30"].copy()
    val = df[(df["date"] > "2024-06-30") & (df["date"] <= "2025-01-31")].copy()
    test = df[df["date"] > "2025-01-31"].copy()

    X_train = train[non_null_cols].fillna(0)
    y_train = train["elevated"]
    X_val = val[non_null_cols].fillna(0)
    y_val = val["elevated"]
    X_test = test[non_null_cols].fillna(0)
    y_test = test["elevated"]

    print(f"Train: {len(train)} ({y_train.mean():.3f} elevated)")
    print(f"Val: {len(val)} ({y_val.mean():.3f} elevated)")
    print(f"Test: {len(test)} ({y_test.mean():.3f} elevated)")

    return (
        X_train, y_train, X_val, y_val, X_test, y_test,
        non_null_cols, train, val, test, df,
    )


def evaluate(model, X, y, prefix=""):
    y_pred = model.predict(X)
    return {
        f"{prefix}binary_f1": f1_score(y, y_pred, zero_division=0),
        f"{prefix}precision": precision_score(y, y_pred, zero_division=0),
        f"{prefix}recall": recall_score(y, y_pred, zero_division=0),
        f"{prefix}fnr": 1 - recall_score(y, y_pred, zero_division=0),
    }


def ordinal_accuracy(model, X, y_binary, df_subset, prefix=""):
    if len(X) != len(df_subset):
        return {f"{prefix}ordinal_acc_within_1": float("nan")}
    y_pred_binary = model.predict(X)
    actual_levels = df_subset["danger_level"].values
    pred_levels = np.where(y_pred_binary == 1, 3, 2)
    within_1 = np.mean(np.abs(actual_levels - pred_levels) <= 1)
    return {f"{prefix}ordinal_acc_within_1": within_1}


def log_experiment(name, model, X_train, y_train, X_val, y_val, X_test, y_test,
                   feature_names, params, train_df, val_df, test_df):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)

        train_m = evaluate(model, X_train, y_train, "train_")
        val_m = evaluate(model, X_val, y_val, "val_")
        test_m = evaluate(model, X_test, y_test, "test_")

        train_ord = ordinal_accuracy(model, X_train, y_train, train_df, "train_")
        val_ord = ordinal_accuracy(model, X_val, y_val, val_df, "val_")
        test_ord = ordinal_accuracy(model, X_test, y_test, test_df, "test_")

        all_metrics = {**train_m, **val_m, **test_m, **train_ord, **val_ord, **test_ord}
        mlflow.log_metrics(all_metrics)

        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        top_features = {feature_names[i]: float(importances[i]) for i in top_idx}
        mlflow.log_dict(top_features, "top_10_features.json")

        val_cm = confusion_matrix(y_val, model.predict(X_val)).tolist()
        test_cm = confusion_matrix(y_test, model.predict(X_test)).tolist()
        mlflow.log_dict({"val": val_cm, "test": test_cm}, "confusion_matrices.json")

        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"  Train F1={train_m['train_binary_f1']:.3f}  Val F1={val_m['val_binary_f1']:.3f}  Test F1={test_m['test_binary_f1']:.3f}")
        print(f"  Val FNR={val_m['val_fnr']:.3f}  Test FNR={test_m['test_fnr']:.3f}")
        print(f"  Val OrdAcc={val_ord['val_ordinal_acc_within_1']:.3f}  Test OrdAcc={test_ord['test_ordinal_acc_within_1']:.3f}")
        print(f"  Top 3 features: {list(top_features.keys())[:3]}")

        return all_metrics


def main():
    (X_train, y_train, X_val, y_val, X_test, y_test,
     non_null_cols, train_df, val_df, test_df, _full_df) = load_data()

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow/mlflow.db")
    mlflow.set_experiment("rf-hyperparameter-search")

    results = []

    # --- Experiment 1: Regularized RF ---
    print("\n>>> Exp 1: Regularized RF")
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X_train, y_train)
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_sm, y_sm)
    m = log_experiment(
        "1_regularized_rf", clf, X_sm, y_sm, X_val, y_val, X_test, y_test,
        list(X_train.columns), {
            "experiment": "regularized_rf", "n_estimators": 300,
            "max_depth": 15, "min_samples_leaf": 20, "smote": True,
            "class_weight": "balanced",
        }, train_df, val_df, test_df,
    )
    results.append(("1_regularized_rf", m))

    # --- Experiment 2: Heavily regularized RF ---
    print("\n>>> Exp 2: Heavily regularized RF")
    clf2 = RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=50,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf2.fit(X_sm, y_sm)
    m = log_experiment(
        "2_heavy_regularized_rf", clf2, X_sm, y_sm, X_val, y_val, X_test, y_test,
        list(X_train.columns), {
            "experiment": "heavy_regularized_rf", "n_estimators": 500,
            "max_depth": 8, "min_samples_leaf": 50, "smote": True,
            "class_weight": "balanced",
        }, train_df, val_df, test_df,
    )
    results.append(("2_heavy_regularized_rf", m))

    # --- Experiment 3: No SMOTE, cost-sensitive only ---
    print("\n>>> Exp 3: No SMOTE, cost-sensitive only")
    clf3 = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=20,
        class_weight={0: 1, 1: 5}, random_state=42, n_jobs=-1,
    )
    clf3.fit(X_train, y_train)
    m = log_experiment(
        "3_no_smote_cost_sensitive", clf3, X_train, y_train, X_val, y_val, X_test, y_test,
        list(X_train.columns), {
            "experiment": "no_smote_cost_sensitive", "n_estimators": 300,
            "max_depth": 15, "min_samples_leaf": 20, "smote": False,
            "class_weight": "{0:1, 1:5}",
        }, train_df, val_df, test_df,
    )
    results.append(("3_no_smote_cost_sensitive", m))

    # --- Experiment 4: Aggressive FNR penalty ---
    print("\n>>> Exp 4: Aggressive FNR penalty")
    clf4 = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=20,
        class_weight={0: 1, 1: 10}, random_state=42, n_jobs=-1,
    )
    clf4.fit(X_train, y_train)
    m = log_experiment(
        "4_aggressive_fnr_penalty", clf4, X_train, y_train, X_val, y_val, X_test, y_test,
        list(X_train.columns), {
            "experiment": "aggressive_fnr_penalty", "n_estimators": 300,
            "max_depth": 12, "min_samples_leaf": 20, "smote": False,
            "class_weight": "{0:1, 1:10}",
        }, train_df, val_df, test_df,
    )
    results.append(("4_aggressive_fnr_penalty", m))

    # --- Experiment 5: Top-30 features only ---
    print("\n>>> Exp 5: Top-30 features only")
    baseline_clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    baseline_clf.fit(X_train, y_train)
    importances = baseline_clf.feature_importances_
    top30_idx = np.argsort(importances)[::-1][:30]
    top30_cols = [non_null_cols[i] for i in top30_idx]
    print(f"  Top 30 features: {top30_cols[:5]}...")

    sm5 = SMOTE(random_state=42)
    X_train_30 = X_train[top30_cols]
    X_sm5, y_sm5 = sm5.fit_resample(X_train_30, y_train)
    clf5 = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf5.fit(X_sm5, y_sm5)
    m = log_experiment(
        "5_top30_features", clf5, X_sm5, y_sm5,
        X_val[top30_cols], y_val, X_test[top30_cols], y_test,
        top30_cols, {
            "experiment": "top30_features", "n_estimators": 300,
            "max_depth": 15, "min_samples_leaf": 20, "smote": True,
            "class_weight": "balanced", "n_features": 30,
        }, train_df, val_df, test_df,
    )
    results.append(("5_top30_features", m))

    # --- Experiment 6: Physics features only ---
    print("\n>>> Exp 6: Physics features only")
    physics_avail = [c for c in PHYSICS_FEATURES if c in X_train.columns]
    print(f"  Physics features ({len(physics_avail)}): {physics_avail}")
    sm6 = SMOTE(random_state=42)
    X_train_phys = X_train[physics_avail]
    X_sm6, y_sm6 = sm6.fit_resample(X_train_phys, y_train)
    clf6 = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf6.fit(X_sm6, y_sm6)
    m = log_experiment(
        "6_physics_only", clf6, X_sm6, y_sm6,
        X_val[physics_avail], y_val, X_test[physics_avail], y_test,
        physics_avail, {
            "experiment": "physics_only", "n_estimators": 300,
            "max_depth": 15, "min_samples_leaf": 20, "smote": True,
            "class_weight": "balanced", "n_features": len(physics_avail),
        }, train_df, val_df, test_df,
    )
    results.append(("6_physics_only", m))

    # --- Experiment 7: Weather features only ---
    print("\n>>> Exp 7: Weather features only")
    weather_avail = [c for c in get_feature_columns() if c in non_null_cols]
    print(f"  Weather features: {len(weather_avail)}")
    sm7 = SMOTE(random_state=42)
    X_train_wx = X_train[weather_avail]
    X_sm7, y_sm7 = sm7.fit_resample(X_train_wx, y_train)
    clf7 = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf7.fit(X_sm7, y_sm7)
    m = log_experiment(
        "7_weather_only", clf7, X_sm7, y_sm7,
        X_val[weather_avail], y_val, X_test[weather_avail], y_test,
        weather_avail, {
            "experiment": "weather_only", "n_estimators": 300,
            "max_depth": 15, "min_samples_leaf": 20, "smote": True,
            "class_weight": "balanced", "n_features": len(weather_avail),
        }, train_df, val_df, test_df,
    )
    results.append(("7_weather_only", m))

    # --- Experiment 8: Seed variance on best regularized config ---
    print("\n>>> Exp 8: Seed variance (regularized RF config)")
    seed_results = []
    for seed in [42, 123, 456, 789, 2024]:
        sm8 = SMOTE(random_state=seed)
        X_sm8, y_sm8 = sm8.fit_resample(X_train, y_train)
        clf8 = RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=20,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
        clf8.fit(X_sm8, y_sm8)
        m = log_experiment(
            f"8_seed_{seed}", clf8, X_sm8, y_sm8, X_val, y_val, X_test, y_test,
            list(X_train.columns), {
                "experiment": f"seed_variance_{seed}", "n_estimators": 300,
                "max_depth": 15, "min_samples_leaf": 20, "smote": True,
                "class_weight": "balanced", "seed": seed,
            }, train_df, val_df, test_df,
        )
        seed_results.append(m)
        results.append((f"8_seed_{seed}", m))

    val_f1s = [m["val_binary_f1"] for m in seed_results]
    val_fnrs = [m["val_fnr"] for m in seed_results]
    print("\n  Seed variance summary:")
    print(f"    Val F1:  mean={np.mean(val_f1s):.3f} std={np.std(val_f1s):.3f}")
    print(f"    Val FNR: mean={np.mean(val_fnrs):.3f} std={np.std(val_fnrs):.3f}")

    # --- Experiment 9: 3-fold temporal cross-validation ---
    print("\n>>> Exp 9: 3-fold temporal CV")
    train_dates = train_df["date"].sort_values().unique()
    n = len(train_dates)
    fold_size = n // 4
    cv_metrics = []

    for fold_i in range(3):
        fold_train_end = train_dates[fold_size * (fold_i + 1)]
        fold_val_start = train_dates[fold_size * (fold_i + 1)]
        fold_val_end = train_dates[min(fold_size * (fold_i + 2), n - 1)]

        fold_train_mask = train_df["date"] <= pd.Timestamp(fold_train_end)
        fold_val_mask = (train_df["date"] > pd.Timestamp(fold_val_start)) & (train_df["date"] <= pd.Timestamp(fold_val_end))

        X_ft = X_train[fold_train_mask]
        y_ft = y_train[fold_train_mask]
        X_fv = X_train[fold_val_mask]
        y_fv = y_train[fold_val_mask]

        if len(X_ft) == 0 or len(X_fv) == 0:
            continue

        sm9 = SMOTE(random_state=42)
        X_ft_sm, y_ft_sm = sm9.fit_resample(X_ft, y_ft)
        clf9 = RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        clf9.fit(X_ft_sm, y_ft_sm)

        fold_train_metrics = evaluate(clf9, X_ft_sm, y_ft_sm, "train_")
        fold_val_metrics = evaluate(clf9, X_fv, y_fv, "val_")

        with mlflow.start_run(run_name=f"9_temporal_cv_fold_{fold_i}"):
            mlflow.log_params({
                "experiment": f"temporal_cv_fold_{fold_i}",
                "fold": fold_i,
                "fold_train_size": len(X_ft),
                "fold_val_size": len(X_fv),
                "n_estimators": 300, "max_depth": 15,
                "min_samples_leaf": 20,
            })
            mlflow.log_metrics({**fold_train_metrics, **fold_val_metrics})

        cv_metrics.append(fold_val_metrics)
        print(f"  Fold {fold_i}: Val F1={fold_val_metrics['val_binary_f1']:.3f} Val FNR={fold_val_metrics['val_fnr']:.3f}")

    if cv_metrics:
        cv_f1s = [m["val_binary_f1"] for m in cv_metrics]
        cv_fnrs = [m["val_fnr"] for m in cv_metrics]
        print(f"  CV mean: Val F1={np.mean(cv_f1s):.3f} +/- {np.std(cv_f1s):.3f}")
        print(f"  CV mean: Val FNR={np.mean(cv_fnrs):.3f} +/- {np.std(cv_fnrs):.3f}")
        results.append(("9_temporal_cv_mean", {
            "val_binary_f1": np.mean(cv_f1s),
            "val_fnr": np.mean(cv_fnrs),
            "train_binary_f1": np.mean([m["train_binary_f1"] for m in [evaluate(clf9, X_ft_sm, y_ft_sm, "train_")]]),
            "test_binary_f1": float("nan"),
            "test_fnr": float("nan"),
        }))

    # === RANKINGS ===
    print("\n" + "=" * 80)
    print("EXPERIMENT RANKINGS (by Val Binary F1)")
    print("=" * 80)
    ranked_f1 = sorted(results, key=lambda x: x[1].get("val_binary_f1", 0), reverse=True)
    print(f"{'Rank':<5} {'Experiment':<30} {'Train F1':<10} {'Val F1':<10} {'Test F1':<10} {'Val FNR':<10} {'Test FNR':<10}")
    print("-" * 85)
    for i, (name, m) in enumerate(ranked_f1):
        train_f1 = m.get("train_binary_f1", float("nan"))
        val_f1 = m.get("val_binary_f1", float("nan"))
        test_f1 = m.get("test_binary_f1", float("nan"))
        val_fnr = m.get("val_fnr", float("nan"))
        test_fnr = m.get("test_fnr", float("nan"))
        print(f"{i+1:<5} {name:<30} {train_f1:<10.3f} {val_f1:<10.3f} {test_f1:<10.3f} {val_fnr:<10.3f} {test_fnr:<10.3f}")

    print("\n" + "=" * 80)
    print("EXPERIMENT RANKINGS (by Val FNR, lower is better)")
    print("=" * 80)
    ranked_fnr = sorted(results, key=lambda x: x[1].get("val_fnr", 1.0))
    print(f"{'Rank':<5} {'Experiment':<30} {'Val FNR':<10} {'Test FNR':<10} {'Val F1':<10} {'Test F1':<10} {'Val Prec':<10} {'Val Rec':<10}")
    print("-" * 95)
    for i, (name, m) in enumerate(ranked_fnr):
        val_fnr = m.get("val_fnr", float("nan"))
        test_fnr = m.get("test_fnr", float("nan"))
        val_f1 = m.get("val_binary_f1", float("nan"))
        test_f1 = m.get("test_binary_f1", float("nan"))
        val_prec = m.get("val_precision", float("nan"))
        val_rec = m.get("val_recall", float("nan"))
        print(f"{i+1:<5} {name:<30} {val_fnr:<10.3f} {test_fnr:<10.3f} {val_f1:<10.3f} {test_f1:<10.3f} {val_prec:<10.3f} {val_rec:<10.3f}")

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    core_results = [(n, m) for n, m in results if not n.startswith("8_seed_") and not n.startswith("9_")]
    best_balanced = min(core_results, key=lambda x: x[1].get("val_fnr", 1.0) - x[1].get("val_binary_f1", 0) * 0.5)
    best_f1 = max(core_results, key=lambda x: x[1].get("val_binary_f1", 0))
    best_fnr = min(core_results, key=lambda x: x[1].get("val_fnr", 1.0))
    print(f"Best Val F1: {best_f1[0]} (F1={best_f1[1]['val_binary_f1']:.3f}, FNR={best_f1[1]['val_fnr']:.3f})")
    print(f"Best Val FNR: {best_fnr[0]} (FNR={best_fnr[1]['val_fnr']:.3f}, F1={best_fnr[1]['val_binary_f1']:.3f})")
    print(f"Best Balanced: {best_balanced[0]} (F1={best_balanced[1]['val_binary_f1']:.3f}, FNR={best_balanced[1]['val_fnr']:.3f})")

    overfitting = [(n, m["train_binary_f1"] - m["val_binary_f1"]) for n, m in core_results if "train_binary_f1" in m]
    print("\nOverfitting gap (train-val F1):")
    for name, gap in sorted(overfitting, key=lambda x: x[1]):
        print(f"  {name:<30} {gap:+.3f}")


if __name__ == "__main__":
    main()
