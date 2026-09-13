"""Round 2 experiments: threshold tuning, boosting models, ensemble."""

import warnings

import duckdb
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

import mlflow
from avalanche_ml.features.physics import PHYSICS_FEATURES
from avalanche_ml.features.weather import get_feature_columns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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


def evaluate_predictions(y_true, y_pred, y_proba, prefix=""):
    metrics = {
        f"{prefix}binary_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}fnr": 1 - recall_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics[f"{prefix}brier_score"] = brier_score_loss(y_true, y_proba)
    return metrics


def ordinal_accuracy(y_pred_binary, df_subset, prefix=""):
    actual_levels = df_subset["danger_level"].values
    pred_levels = np.where(y_pred_binary == 1, 3, 2)
    within_1 = np.mean(np.abs(actual_levels - pred_levels) <= 1)
    return {f"{prefix}ordinal_acc_within_1": within_1}


def log_experiment(name, y_preds, y_probas, y_trues, dfs, feature_names,
                   feature_importances, params):
    (y_train, y_val, y_test) = y_trues
    (yp_train, yp_val, yp_test) = y_preds
    (prob_train, prob_val, prob_test) = y_probas
    (train_df, val_df, test_df) = dfs

    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)

        train_m = evaluate_predictions(y_train, yp_train, prob_train, "train_")
        val_m = evaluate_predictions(y_val, yp_val, prob_val, "val_")
        test_m = evaluate_predictions(y_test, yp_test, prob_test, "test_")

        train_ord = ordinal_accuracy(yp_train, train_df, "train_")
        val_ord = ordinal_accuracy(yp_val, val_df, "val_")
        test_ord = ordinal_accuracy(yp_test, test_df, "test_")

        all_metrics = {**train_m, **val_m, **test_m, **train_ord, **val_ord, **test_ord}
        mlflow.log_metrics(all_metrics)

        if feature_importances is not None and feature_names is not None:
            top_idx = np.argsort(feature_importances)[::-1][:10]
            top_features = {feature_names[i]: float(feature_importances[i]) for i in top_idx}
            mlflow.log_dict(top_features, "top_10_features.json")
        else:
            top_features = {}

        val_cm = confusion_matrix(y_val, yp_val).tolist()
        test_cm = confusion_matrix(y_test, yp_test).tolist()
        mlflow.log_dict({"val": val_cm, "test": test_cm}, "confusion_matrices.json")

        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"  Train F1={train_m['train_binary_f1']:.3f}  Val F1={val_m['val_binary_f1']:.3f}  Test F1={test_m['test_binary_f1']:.3f}")
        print(f"  Val FNR={val_m['val_fnr']:.3f}  Test FNR={test_m['test_fnr']:.3f}")
        print(f"  Val Prec={val_m['val_precision']:.3f}  Val Rec={val_m['val_recall']:.3f}")
        if prob_val is not None:
            print(f"  Val Brier={val_m['val_brier_score']:.4f}  Test Brier={test_m['test_brier_score']:.4f}")
        print(f"  Val OrdAcc={val_ord['val_ordinal_acc_within_1']:.3f}  Test OrdAcc={test_ord['test_ordinal_acc_within_1']:.3f}")
        if top_features:
            print(f"  Top 3 features: {list(top_features.keys())[:3]}")

        return all_metrics


def predict_with_threshold(model, X, threshold):
    proba = model.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int), proba


def main():
    (X_train, y_train, X_val, y_val, X_test, y_test,
     _non_null_cols, train_df, val_df, test_df, _full_df) = load_data()

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow/mlflow.db")
    mlflow.set_experiment("rf-round2-experiments")

    results = []
    feature_names = list(X_train.columns)

    # =========================================================================
    # Exp 10-12: Intermediate class weights
    # =========================================================================
    for exp_num, weight in [(10, 6), (11, 7), (12, 8)]:
        print(f"\n>>> Exp {exp_num}: Class weight {{0:1, 1:{weight}}}")
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=30,
            class_weight={0: 1, 1: weight}, random_state=42, n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        yp_train, prob_train = predict_with_threshold(clf, X_train, 0.5)
        yp_val, prob_val = predict_with_threshold(clf, X_val, 0.5)
        yp_test, prob_test = predict_with_threshold(clf, X_test, 0.5)

        m = log_experiment(
            f"{exp_num}_weight_1_{weight}", (yp_train, yp_val, yp_test),
            (prob_train, prob_val, prob_test), (y_train, y_val, y_test),
            (train_df, val_df, test_df), feature_names, clf.feature_importances_,
            {"experiment": f"weight_1_{weight}", "n_estimators": 300,
             "max_depth": 12, "min_samples_leaf": 30, "smote": False,
             "class_weight": f"{{0:1, 1:{weight}}}"},
        )
        results.append((f"{exp_num}_weight_1_{weight}", m))

    # =========================================================================
    # Exp 13-15: Threshold tuning on exp 3 config
    # =========================================================================
    print("\n>>> Training base model for threshold experiments (exp 3 config)")
    base_rf = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=20,
        class_weight={0: 1, 1: 5}, random_state=42, n_jobs=-1,
    )
    base_rf.fit(X_train, y_train)

    for exp_num, threshold in [(13, 0.35), (14, 0.30), (15, 0.25)]:
        print(f"\n>>> Exp {exp_num}: Threshold={threshold}")
        yp_train, prob_train = predict_with_threshold(base_rf, X_train, threshold)
        yp_val, prob_val = predict_with_threshold(base_rf, X_val, threshold)
        yp_test, prob_test = predict_with_threshold(base_rf, X_test, threshold)

        m = log_experiment(
            f"{exp_num}_threshold_{threshold}", (yp_train, yp_val, yp_test),
            (prob_train, prob_val, prob_test), (y_train, y_val, y_test),
            (train_df, val_df, test_df), feature_names, base_rf.feature_importances_,
            {"experiment": f"threshold_{threshold}", "base_config": "exp3_cost_sensitive",
             "n_estimators": 300, "max_depth": 15, "min_samples_leaf": 20,
             "smote": False, "class_weight": "{0:1, 1:5}",
             "threshold": threshold},
        )
        results.append((f"{exp_num}_threshold_{threshold}", m))

    # =========================================================================
    # Exp 16: Probability calibration
    # =========================================================================
    print("\n>>> Exp 16: Probability calibration (isotonic)")
    uncalibrated_proba_val = base_rf.predict_proba(X_val)[:, 1]
    uncalibrated_brier = brier_score_loss(y_val, uncalibrated_proba_val)
    print(f"  Uncalibrated Brier score (val): {uncalibrated_brier:.4f}")

    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(uncalibrated_proba_val, y_val)

    raw_prob_train = base_rf.predict_proba(X_train)[:, 1]
    cal_prob_train = iso_reg.predict(raw_prob_train)
    cal_prob_val = iso_reg.predict(uncalibrated_proba_val)
    cal_prob_test = iso_reg.predict(base_rf.predict_proba(X_test)[:, 1])

    yp_train_cal = (cal_prob_train >= 0.5).astype(int)
    yp_val_cal = (cal_prob_val >= 0.5).astype(int)
    yp_test_cal = (cal_prob_test >= 0.5).astype(int)

    calibrated_brier = brier_score_loss(y_val, cal_prob_val)
    print(f"  Calibrated Brier score (val): {calibrated_brier:.4f}")
    print(f"  Brier improvement: {uncalibrated_brier - calibrated_brier:+.4f}")

    m = log_experiment(
        "16_calibrated", (yp_train_cal, yp_val_cal, yp_test_cal),
        (cal_prob_train, cal_prob_val, cal_prob_test), (y_train, y_val, y_test),
        (train_df, val_df, test_df), feature_names, base_rf.feature_importances_,
        {"experiment": "calibrated_isotonic", "base_config": "exp3_cost_sensitive",
         "calibration": "isotonic", "uncalibrated_brier": round(uncalibrated_brier, 4),
         "calibrated_brier": round(calibrated_brier, 4)},
    )
    results.append(("16_calibrated", m))

    # =========================================================================
    # Exp 17: XGBoost
    # =========================================================================
    print("\n>>> Exp 17: XGBoost")
    xgb = XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        scale_pos_weight=5, early_stopping_rounds=20,
        eval_metric="logloss", random_state=42,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  XGBoost stopped at {xgb.best_iteration} iterations")

    yp_train_xgb, prob_train_xgb = predict_with_threshold(xgb, X_train, 0.5)
    yp_val_xgb, prob_val_xgb = predict_with_threshold(xgb, X_val, 0.5)
    yp_test_xgb, prob_test_xgb = predict_with_threshold(xgb, X_test, 0.5)

    m = log_experiment(
        "17_xgboost", (yp_train_xgb, yp_val_xgb, yp_test_xgb),
        (prob_train_xgb, prob_val_xgb, prob_test_xgb), (y_train, y_val, y_test),
        (train_df, val_df, test_df), feature_names, xgb.feature_importances_,
        {"experiment": "xgboost", "n_estimators": 500, "max_depth": 8,
         "learning_rate": 0.05, "scale_pos_weight": 5,
         "early_stopping_rounds": 20, "best_iteration": xgb.best_iteration},
    )
    results.append(("17_xgboost", m))

    # =========================================================================
    # Exp 18: LightGBM
    # =========================================================================
    print("\n>>> Exp 18: LightGBM")
    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        scale_pos_weight=5, metric="binary_logloss",
        random_state=42, verbose=-1,
    )
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            __import__("lightgbm").early_stopping(20, verbose=False),
            __import__("lightgbm").log_evaluation(period=0),
        ],
    )
    print(f"  LightGBM stopped at {lgbm.best_iteration_} iterations")

    yp_train_lgbm, prob_train_lgbm = predict_with_threshold(lgbm, X_train, 0.5)
    yp_val_lgbm, prob_val_lgbm = predict_with_threshold(lgbm, X_val, 0.5)
    yp_test_lgbm, prob_test_lgbm = predict_with_threshold(lgbm, X_test, 0.5)

    m = log_experiment(
        "18_lightgbm", (yp_train_lgbm, yp_val_lgbm, yp_test_lgbm),
        (prob_train_lgbm, prob_val_lgbm, prob_test_lgbm), (y_train, y_val, y_test),
        (train_df, val_df, test_df), feature_names, lgbm.feature_importances_,
        {"experiment": "lightgbm", "n_estimators": 500, "max_depth": 8,
         "learning_rate": 0.05, "scale_pos_weight": 5,
         "best_iteration": lgbm.best_iteration_},
    )
    results.append(("18_lightgbm", m))

    # =========================================================================
    # Exp 19: XGBoost + threshold tuning
    # =========================================================================
    print("\n>>> Exp 19: XGBoost + threshold=0.30")
    yp_train_xgbt, prob_train_xgbt = predict_with_threshold(xgb, X_train, 0.30)
    yp_val_xgbt, prob_val_xgbt = predict_with_threshold(xgb, X_val, 0.30)
    yp_test_xgbt, prob_test_xgbt = predict_with_threshold(xgb, X_test, 0.30)

    m = log_experiment(
        "19_xgboost_threshold_0.30", (yp_train_xgbt, yp_val_xgbt, yp_test_xgbt),
        (prob_train_xgbt, prob_val_xgbt, prob_test_xgbt), (y_train, y_val, y_test),
        (train_df, val_df, test_df), feature_names, xgb.feature_importances_,
        {"experiment": "xgboost_threshold", "threshold": 0.30,
         "n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
         "scale_pos_weight": 5, "best_iteration": xgb.best_iteration},
    )
    results.append(("19_xgboost_threshold_0.30", m))

    # =========================================================================
    # Exp 20: Ensemble (RF + XGBoost + LightGBM)
    # =========================================================================
    print("\n>>> Exp 20: Ensemble (RF + XGBoost + LightGBM)")
    ens_prob_train = (prob_train + prob_train_xgb + prob_train_lgbm) / 3
    ens_prob_val = (prob_val + prob_val_xgb + prob_val_lgbm) / 3
    ens_prob_test = (prob_test + prob_test_xgb + prob_test_lgbm) / 3

    yp_train_ens = (ens_prob_train >= 0.5).astype(int)
    yp_val_ens = (ens_prob_val >= 0.5).astype(int)
    yp_test_ens = (ens_prob_test >= 0.5).astype(int)

    m = log_experiment(
        "20_ensemble", (yp_train_ens, yp_val_ens, yp_test_ens),
        (ens_prob_train, ens_prob_val, ens_prob_test), (y_train, y_val, y_test),
        (train_df, val_df, test_df), None, None,
        {"experiment": "ensemble_rf_xgb_lgbm", "models": "rf_exp3+xgb+lgbm",
         "aggregation": "mean_proba", "threshold": 0.5},
    )
    results.append(("20_ensemble", m))

    # =========================================================================
    # Summary tables
    # =========================================================================
    print("\n" + "=" * 100)
    print("ROUND 2 RESULTS — SORTED BY VAL F1")
    print("=" * 100)
    ranked_f1 = sorted(results, key=lambda x: x[1].get("val_binary_f1", 0), reverse=True)
    header = f"{'Rank':<5} {'Experiment':<30} {'Train F1':<10} {'Val F1':<10} {'Test F1':<10} {'Val FNR':<10} {'Test FNR':<10} {'Val Brier':<10}"
    print(header)
    print("-" * len(header))
    for i, (name, m) in enumerate(ranked_f1):
        brier = m.get("val_brier_score", float("nan"))
        print(f"{i+1:<5} {name:<30} {m.get('train_binary_f1', float('nan')):<10.3f} "
              f"{m.get('val_binary_f1', float('nan')):<10.3f} {m.get('test_binary_f1', float('nan')):<10.3f} "
              f"{m.get('val_fnr', float('nan')):<10.3f} {m.get('test_fnr', float('nan')):<10.3f} "
              f"{brier:<10.4f}")

    print("\n" + "=" * 100)
    print("ROUND 2 RESULTS — SORTED BY VAL FNR (lower is better)")
    print("=" * 100)
    ranked_fnr = sorted(results, key=lambda x: x[1].get("val_fnr", 1.0))
    header = f"{'Rank':<5} {'Experiment':<30} {'Val FNR':<10} {'Test FNR':<10} {'Val F1':<10} {'Test F1':<10} {'Val Prec':<10} {'Val Rec':<10}"
    print(header)
    print("-" * len(header))
    for i, (name, m) in enumerate(ranked_fnr):
        print(f"{i+1:<5} {name:<30} {m.get('val_fnr', float('nan')):<10.3f} "
              f"{m.get('test_fnr', float('nan')):<10.3f} {m.get('val_binary_f1', float('nan')):<10.3f} "
              f"{m.get('test_binary_f1', float('nan')):<10.3f} {m.get('val_precision', float('nan')):<10.3f} "
              f"{m.get('val_recall', float('nan')):<10.3f}")

    # Pareto frontier
    print("\n" + "=" * 100)
    print("PARETO FRONTIER (not dominated on both Val F1 and Val FNR)")
    print("=" * 100)
    pareto = []
    for name, m in results:
        f1 = m.get("val_binary_f1", 0)
        fnr = m.get("val_fnr", 1.0)
        dominated = False
        for other_name, other_m in results:
            if other_name == name:
                continue
            other_f1 = other_m.get("val_binary_f1", 0)
            other_fnr = other_m.get("val_fnr", 1.0)
            if other_f1 >= f1 and other_fnr <= fnr and (other_f1 > f1 or other_fnr < fnr):
                dominated = True
                break
        if not dominated:
            pareto.append((name, m))

    header = f"{'Experiment':<30} {'Val F1':<10} {'Val FNR':<10} {'Test F1':<10} {'Test FNR':<10} {'Val Prec':<10} {'Val Rec':<10}"
    print(header)
    print("-" * len(header))
    for name, m in sorted(pareto, key=lambda x: x[1].get("val_binary_f1", 0), reverse=True):
        print(f"{name:<30} {m.get('val_binary_f1', float('nan')):<10.3f} "
              f"{m.get('val_fnr', float('nan')):<10.3f} {m.get('test_binary_f1', float('nan')):<10.3f} "
              f"{m.get('test_fnr', float('nan')):<10.3f} {m.get('val_precision', float('nan')):<10.3f} "
              f"{m.get('val_recall', float('nan')):<10.3f}")

    # Recommendation
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    print("For a safety-critical avalanche prediction system:")
    print()

    best_f1 = max(results, key=lambda x: x[1].get("val_binary_f1", 0))
    best_fnr = min(results, key=lambda x: x[1].get("val_fnr", 1.0))
    print(f"  Best Val F1:  {best_f1[0]} (F1={best_f1[1]['val_binary_f1']:.3f}, FNR={best_f1[1]['val_fnr']:.3f})")
    print(f"  Best Val FNR: {best_fnr[0]} (FNR={best_fnr[1]['val_fnr']:.3f}, F1={best_fnr[1]['val_binary_f1']:.3f})")
    print()

    # Find the Pareto point that minimizes a combined score: FNR * 2 + (1 - F1)
    # This weights FNR twice as heavily as F1 loss, appropriate for safety-critical
    def safety_score(item):
        m = item[1]
        return m.get("val_fnr", 1.0) * 2 + (1 - m.get("val_binary_f1", 0))

    recommended = min(pareto, key=safety_score) if pareto else min(results, key=safety_score)
    print(f"  RECOMMENDED: {recommended[0]}")
    print(f"    Val F1={recommended[1]['val_binary_f1']:.3f}, Val FNR={recommended[1]['val_fnr']:.3f}")
    print(f"    Test F1={recommended[1]['test_binary_f1']:.3f}, Test FNR={recommended[1]['test_fnr']:.3f}")
    print("    Reasoning: FNR weighted 2x in safety-critical scoring.")
    print("    Missing a dangerous condition (FNR) is worse than a false alarm (low precision).")

    # Overfitting check
    print("\n  Overfitting gaps (train-val F1):")
    for name, m in sorted(results, key=lambda x: x[1].get("train_binary_f1", 0) - x[1].get("val_binary_f1", 0)):
        gap = m.get("train_binary_f1", 0) - m.get("val_binary_f1", 0)
        print(f"    {name:<30} {gap:+.3f}")


if __name__ == "__main__":
    main()
