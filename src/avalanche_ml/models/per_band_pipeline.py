from __future__ import annotations

import datetime
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from avalanche_ml.features.alignment import ELEVATION_BANDS, PROBLEM_TYPE_FLAGS
from avalanche_ml.features.physics import PHYSICS_FEATURES
from avalanche_ml.features.weather import get_feature_columns

PROB_COLUMNS = [f"{pt}_prob" for pt in PROBLEM_TYPE_FLAGS]

ENSEMBLE_CONFIGS = [
    {"n_estimators": 200, "max_depth": 12},
    {"n_estimators": 300, "max_depth": 12},
    {"n_estimators": 500, "max_depth": 12},
]

STAGE1_DEFAULTS = {
    "min_samples_leaf": 20,
    "class_weight": {0: 1, 1: 5},
    "random_state": 42,
}

STAGE2_DEFAULTS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 20,
    "random_state": 42,
}

DANGER_CLASSES = [1, 2, 3, 4]


def _feature_columns() -> list[str]:
    return get_feature_columns() + list(PHYSICS_FEATURES)


def staged_chronological_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dates = pd.to_datetime(df["date"])
    sorted_dates = dates.sort_values()
    n = len(sorted_dates)

    p40 = sorted_dates.iloc[int(n * 0.40)]
    p50 = sorted_dates.iloc[int(n * 0.50)]
    p85 = sorted_dates.iloc[int(n * 0.85)]
    p925 = sorted_dates.iloc[int(n * 0.925)]

    return {
        "s1_train": df[dates <= p40].copy(),
        "s1_eval": df[(dates > p40) & (dates <= p50)].copy(),
        "s2_train": df[(dates > p40) & (dates <= p85)].copy(),
        "s2_val": df[(dates > p85) & (dates <= p925)].copy(),
        "s2_test": df[dates > p925].copy(),
    }


def _train_stage1_cell(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    configs: list[dict],
    base_params: dict,
) -> dict:
    trained = []
    for cfg in configs:
        params = {**base_params, **cfg, "n_jobs": -1}
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)

        if len(np.unique(y_eval)) < 2:
            score = 0.0
        else:
            preds = clf.predict(X_eval)
            score = float(f1_score(y_eval, preds, average="macro", zero_division=0))

        trained.append({"model": clf, "score": score, "config": cfg})

    trained.sort(key=lambda x: x["score"], reverse=True)
    top_n = min(3, len(trained))
    kept = trained[:top_n]

    return {
        "models": [k["model"] for k in kept],
        "scores": [k["score"] for k in kept],
        "n_configs_trained": len(configs),
        "n_configs_kept": top_n,
    }


def _ensemble_predict_proba(models: list[RandomForestClassifier], X: np.ndarray) -> np.ndarray:
    probas = []
    for clf in models:
        if len(clf.classes_) == 1:
            p = np.full(len(X), float(clf.classes_[0]))
        else:
            p = clf.predict_proba(X)[:, 1]
        probas.append(p)
    return np.mean(probas, axis=0)


class PerBandPipeline:
    def __init__(self):
        self.stage1_ensembles_: dict[str, dict[str, list[RandomForestClassifier]]] = {}
        self.stage2_models_: dict[str, RandomForestClassifier] = {}
        self.feature_names_: list[str] = []
        self.s2_feature_names_: dict[str, list[str]] = {}

    def train(self, df: pd.DataFrame) -> dict:
        feat_cols = _feature_columns()
        available = [c for c in feat_cols if c in df.columns]
        self.feature_names_ = available

        results = {}
        for band in ELEVATION_BANDS:
            band_df = df[df["elevation_band"] == band].copy()
            if len(band_df) < 50:
                continue
            results[band] = self._train_band(band, band_df, available)

        return results

    def _train_band(self, band: str, band_df: pd.DataFrame, feat_cols: list[str]) -> dict:
        splits = staged_chronological_split(band_df)

        X_s1_train = splits["s1_train"][feat_cols].values.astype(np.float64)
        X_s1_eval = splits["s1_eval"][feat_cols].values.astype(np.float64)
        np.nan_to_num(X_s1_train, copy=False, nan=0.0)
        np.nan_to_num(X_s1_eval, copy=False, nan=0.0)

        ensembles = {}
        ensemble_info = {}
        for pt in PROBLEM_TYPE_FLAGS:
            y_train = splits["s1_train"][pt].values.astype(int)
            y_eval = splits["s1_eval"][pt].values.astype(int)

            cell = _train_stage1_cell(
                X_s1_train, y_train, X_s1_eval, y_eval,
                ENSEMBLE_CONFIGS, STAGE1_DEFAULTS,
            )
            ensembles[pt] = cell["models"]
            ensemble_info[pt] = {
                "n_configs_trained": cell["n_configs_trained"],
                "n_configs_kept": cell["n_configs_kept"],
                "scores": cell["scores"],
            }

        self.stage1_ensembles_[band] = ensembles

        X_s2_train = splits["s2_train"][feat_cols].values.astype(np.float64)
        np.nan_to_num(X_s2_train, copy=False, nan=0.0)

        s1_probs_train = {}
        for pt in PROBLEM_TYPE_FLAGS:
            s1_probs_train[f"{pt}_prob"] = _ensemble_predict_proba(
                ensembles[pt], X_s2_train
            )

        s1_prob_arr = np.column_stack([s1_probs_train[c] for c in PROB_COLUMNS])
        X_s2_full = np.hstack([X_s2_train, s1_prob_arr])

        s2_feature_names = feat_cols + PROB_COLUMNS
        self.s2_feature_names_[band] = s2_feature_names

        y_s2_train = splits["s2_train"]["danger_level"].values.astype(int)
        y_s2_train = np.clip(y_s2_train, 1, 4)

        clf = RandomForestClassifier(
            **STAGE2_DEFAULTS,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf.fit(X_s2_full, y_s2_train)
        self.stage2_models_[band] = clf

        val_metrics = self._evaluate_stage2(
            band, splits["s2_val"], feat_cols
        )
        test_metrics = self._evaluate_stage2(
            band, splits["s2_test"], feat_cols
        )

        s1_val_metrics = self._evaluate_stage1(
            band, splits["s1_eval"], feat_cols
        )

        return {
            "stage1_ensembles": ensemble_info,
            "stage1_eval_metrics": s1_val_metrics,
            "stage2_val_metrics": val_metrics,
            "stage2_test_metrics": test_metrics,
            "s2_feature_columns": s2_feature_names,
            "split_sizes": {
                k: len(v) for k, v in splits.items()
            },
        }

    def _evaluate_stage1(
        self, band: str, eval_df: pd.DataFrame, feat_cols: list[str]
    ) -> dict:
        X = eval_df[feat_cols].values.astype(np.float64)
        np.nan_to_num(X, copy=False, nan=0.0)

        metrics = {}
        for pt in PROBLEM_TYPE_FLAGS:
            y_true = eval_df[pt].values.astype(int)
            proba = _ensemble_predict_proba(self.stage1_ensembles_[band][pt], X)
            y_pred = (proba >= 0.30).astype(int)

            if len(np.unique(y_true)) < 2:
                metrics[pt] = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
                continue

            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            metrics[pt] = {"f1": f1, "precision": prec, "recall": rec}

        return metrics

    def _evaluate_stage2(
        self, band: str, eval_df: pd.DataFrame, feat_cols: list[str]
    ) -> dict:
        if len(eval_df) == 0:
            return {"macro_f1": 0.0, "n_samples": 0}

        X = eval_df[feat_cols].values.astype(np.float64)
        np.nan_to_num(X, copy=False, nan=0.0)

        s1_probs = {}
        for pt in PROBLEM_TYPE_FLAGS:
            s1_probs[f"{pt}_prob"] = _ensemble_predict_proba(
                self.stage1_ensembles_[band][pt], X
            )
        s1_arr = np.column_stack([s1_probs[c] for c in PROB_COLUMNS])
        X_full = np.hstack([X, s1_arr])

        y_true = eval_df["danger_level"].values.astype(int)
        y_true = np.clip(y_true, 1, 4)

        y_pred = self.stage2_models_[band].predict(X_full)

        present = sorted(set(y_true) | set(y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))

        per_class = {}
        for cls in DANGER_CLASSES:
            yt_bin = (y_true == cls).astype(int)
            yp_bin = (y_pred == cls).astype(int)
            if yt_bin.sum() == 0 and yp_bin.sum() == 0:
                per_class[cls] = 0.0
            else:
                per_class[cls] = float(f1_score(yt_bin, yp_bin, zero_division=0))

        y_pred_binary = (y_pred >= 3).astype(int)
        y_true_binary = (y_true >= 3).astype(int)
        fnr = 0.0
        if y_true_binary.sum() > 0:
            fnr = float(((y_true_binary == 1) & (y_pred_binary == 0)).sum() / y_true_binary.sum())

        high_mask = y_true >= 4
        hdr = 1.0
        if high_mask.sum() > 0:
            hdr = float((y_pred[high_mask] >= 3).sum() / high_mask.sum())

        return {
            "macro_f1": macro_f1,
            "per_class_f1": per_class,
            "false_negative_rate": fnr,
            "high_danger_detection_rate": hdr,
            "n_samples": len(y_true),
        }

    def predict(self, df: pd.DataFrame, band: str) -> dict:
        feat_cols = self.feature_names_
        X = df[feat_cols].values.astype(np.float64)
        np.nan_to_num(X, copy=False, nan=0.0)

        s1_probs = {}
        for pt in PROBLEM_TYPE_FLAGS:
            s1_probs[pt] = _ensemble_predict_proba(
                self.stage1_ensembles_[band][pt], X
            )
        s1_arr = np.column_stack([s1_probs[pt] for pt in PROBLEM_TYPE_FLAGS])
        X_full = np.hstack([X, s1_arr])

        y_pred = self.stage2_models_[band].predict(X_full)
        y_proba = self.stage2_models_[band].predict_proba(X_full)

        proba_dist = np.zeros((len(X), 4))
        for i, cls in enumerate(self.stage2_models_[band].classes_):
            idx = int(cls) - 1
            if 0 <= idx < 4:
                proba_dist[:, idx] = y_proba[:, i]
        row_sums = proba_dist.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        proba_dist = proba_dist / row_sums

        return {
            "danger_level": y_pred.tolist(),
            "danger_proba": proba_dist,
            "problem_type_probs": {pt: s1_probs[pt].tolist() for pt in PROBLEM_TYPE_FLAGS},
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        for band in self.stage1_ensembles_:
            band_dir = path / band
            band_dir.mkdir(parents=True, exist_ok=True)

            s1_data = {}
            for pt in PROBLEM_TYPE_FLAGS:
                s1_data[pt] = self.stage1_ensembles_[band][pt]
            joblib.dump(s1_data, band_dir / "stage1_ensembles.joblib")
            joblib.dump(self.stage2_models_[band], band_dir / "stage2_model.joblib")

        meta = {
            "feature_names": self.feature_names_,
            "s2_feature_names": self.s2_feature_names_,
            "bands": list(self.stage1_ensembles_.keys()),
            "saved_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        (path / "meta.json").write_text(json.dumps(meta, default=str))

    @classmethod
    def load(cls, path: Path) -> PerBandPipeline:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())

        instance = cls()
        instance.feature_names_ = meta["feature_names"]
        instance.s2_feature_names_ = meta["s2_feature_names"]

        for band in meta["bands"]:
            band_dir = path / band
            instance.stage1_ensembles_[band] = joblib.load(
                band_dir / "stage1_ensembles.joblib"
            )
            instance.stage2_models_[band] = joblib.load(
                band_dir / "stage2_model.joblib"
            )

        return instance
