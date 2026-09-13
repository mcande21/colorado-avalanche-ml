from __future__ import annotations

import datetime
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

from avalanche_ml.features.alignment import PROBLEM_TYPE_FLAGS


class Stage1RF:
    """Multi-label Random Forest for avalanche problem type prediction.

    Trains one RF per problem type (persistent_slab, storm_slab, loose_wet).
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_leaf: int = 5,
        use_smote: bool = True,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.use_smote = use_smote
        self.random_state = random_state

        self.models_: dict[str, RandomForestClassifier] = {}
        self.feature_names_: list[str] = []
        self.smote_applied_: bool = False
        self.metadata_: dict | None = None

    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self.feature_names_ = list(X.columns)
        X_arr = X.values.astype(np.float64)
        np.nan_to_num(X_arr, copy=False, nan=0.0)

        class_distributions: dict[str, dict[str, int]] = {}
        smote_applied = False

        for pt in PROBLEM_TYPE_FLAGS:
            y_col = y[pt].values.astype(int)
            class_distributions[pt] = {
                "positive": int(y_col.sum()),
                "negative": int(len(y_col) - y_col.sum()),
            }

            X_train, y_train = X_arr, y_col

            if self.use_smote and len(np.unique(y_col)) > 1:
                minority_count = min(np.bincount(y_col))
                if minority_count >= 2:
                    from imblearn.over_sampling import SMOTE
                    n_neighbors = min(5, minority_count - 1)
                    sm = SMOTE(random_state=self.random_state, k_neighbors=n_neighbors)
                    X_train, y_train = sm.fit_resample(X_arr, y_col)
                    smote_applied = True

            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            self.models_[pt] = clf

        self.smote_applied_ = smote_applied
        self.metadata_ = {
            "training_date": datetime.datetime.now(datetime.UTC).isoformat(),
            "feature_names": self.feature_names_,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "use_smote": self.use_smote,
                "random_state": self.random_state,
            },
            "class_distributions": class_distributions,
        }

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        X_arr = X.values.astype(np.float64)
        np.nan_to_num(X_arr, copy=False, nan=0.0)

        results: dict[str, np.ndarray] = {}
        for pt in PROBLEM_TYPE_FLAGS:
            clf = self.models_[pt]
            if len(clf.classes_) == 1:
                prob = np.full(len(X_arr), float(clf.classes_[0]))
            else:
                prob = clf.predict_proba(X_arr)[:, 1]
            results[pt] = prob

        return pd.DataFrame(results, index=X.index)

    def predict_binary(
        self, X: pd.DataFrame, thresholds: dict[str, float]
    ) -> pd.DataFrame:
        probs = self.predict(X)
        binary: dict[str, np.ndarray] = {}
        for pt in PROBLEM_TYPE_FLAGS:
            threshold = thresholds.get(pt, 0.5)
            binary[pt] = (probs[pt] >= threshold).astype(int)
        return pd.DataFrame(binary, index=X.index)

    def shap_values(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        X_arr = X.values.astype(np.float64)
        np.nan_to_num(X_arr, copy=False, nan=0.0)

        result: dict[str, np.ndarray] = {}
        for pt in PROBLEM_TYPE_FLAGS:
            clf = self.models_[pt]
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_arr)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
            result[pt] = sv
        return result

    def global_feature_importance(self) -> dict[str, list[tuple[str, float]]]:
        result: dict[str, list[tuple[str, float]]] = {}
        for pt in PROBLEM_TYPE_FLAGS:
            clf = self.models_[pt]
            importances = clf.feature_importances_
            pairs = list(zip(self.feature_names_, importances.tolist()))
            pairs.sort(key=lambda x: x[1], reverse=True)
            result[pt] = pairs
        return result

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models_, path / "models.joblib")
        meta = {
            "feature_names": self.feature_names_,
            "smote_applied": self.smote_applied_,
            "metadata": self.metadata_,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "use_smote": self.use_smote,
                "random_state": self.random_state,
            },
        }
        (path / "meta.json").write_text(json.dumps(meta, default=str))

    @classmethod
    def load(cls, path: Path) -> Stage1RF:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        hp = meta["hyperparameters"]
        instance = cls(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            min_samples_leaf=hp["min_samples_leaf"],
            use_smote=hp["use_smote"],
            random_state=hp["random_state"],
        )
        instance.models_ = joblib.load(path / "models.joblib")
        instance.feature_names_ = meta["feature_names"]
        instance.smote_applied_ = meta["smote_applied"]
        instance.metadata_ = meta["metadata"]
        return instance
