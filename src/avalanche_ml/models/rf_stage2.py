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
from avalanche_ml.models.rf_stage1 import Stage1RF

PROB_COLUMNS = [f"{pt}_prob" for pt in PROBLEM_TYPE_FLAGS]

# Asymmetric cost: false negative (predicting safe when dangerous) costs more
# Index: [true_elevated][predicted_elevated] — cost[1][0] > cost[0][1]
COST_MATRIX = {
    0: {0: 0, 1: 1},   # true=not-elevated: FP costs 1
    1: {0: 3, 1: 0},   # true=elevated: FN costs 3
}


def _encode_binary(danger_levels: np.ndarray) -> np.ndarray:
    return (danger_levels >= 3).astype(int)


def _encode_3class(danger_levels: np.ndarray) -> np.ndarray:
    result = np.zeros(len(danger_levels), dtype=int)
    result[danger_levels == 2] = 1
    result[danger_levels >= 3] = 2
    return result


def _build_sample_weights(y_binary: np.ndarray) -> np.ndarray:
    weights = np.ones(len(y_binary), dtype=np.float64)
    weights[y_binary == 1] = COST_MATRIX[1][0]
    weights[y_binary == 0] = COST_MATRIX[0][1]
    return weights


class Stage2RF:
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

        self.binary_model_: RandomForestClassifier | None = None
        self.multiclass_model_: RandomForestClassifier | None = None
        self.threeclass_model_: RandomForestClassifier | None = None
        self.feature_names_: list[str] = []
        self.smote_applied_: bool = False
        self.metadata_: dict | None = None

    def get_cost_matrix(self) -> dict[int, dict[int, int]]:
        return COST_MATRIX

    def train(self, X: pd.DataFrame, y_danger: pd.Series | np.ndarray) -> None:
        self.feature_names_ = list(X.columns)
        X_arr = X.values.astype(np.float64)
        np.nan_to_num(X_arr, copy=False, nan=0.0)
        y_arr = np.asarray(y_danger, dtype=int)

        y_binary = _encode_binary(y_arr)
        y_3class = _encode_3class(y_arr)

        smote_applied = False

        X_bin, y_bin = X_arr, y_binary
        if self.use_smote and len(np.unique(y_binary)) > 1:
            minority_count = min(np.bincount(y_binary))
            if minority_count >= 2:
                from imblearn.over_sampling import SMOTE
                n_neighbors = min(5, minority_count - 1)
                sm = SMOTE(random_state=self.random_state, k_neighbors=n_neighbors)
                X_bin, y_bin = sm.fit_resample(X_arr, y_binary)
                smote_applied = True

        sample_weights = _build_sample_weights(y_bin)

        self.binary_model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.binary_model_.fit(X_bin, y_bin, sample_weight=sample_weights)

        self.multiclass_model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.multiclass_model_.fit(X_arr, y_arr)

        self.threeclass_model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.threeclass_model_.fit(X_arr, y_3class)

        self.smote_applied_ = smote_applied
        self.metadata_ = {
            "training_date": datetime.datetime.now(datetime.UTC).isoformat(),
            "feature_names": self.feature_names_,
            "n_samples": len(y_arr),
            "class_distribution_binary": {
                "not_elevated": int((y_binary == 0).sum()),
                "elevated": int((y_binary == 1).sum()),
            },
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "use_smote": self.use_smote,
                "random_state": self.random_state,
            },
        }

    def predict(self, X: pd.DataFrame) -> dict:
        X_arr = X.values.astype(np.float64)
        np.nan_to_num(X_arr, copy=False, nan=0.0)

        danger_binary = self.binary_model_.predict(X_arr)

        danger_proba = self.binary_model_.predict_proba(X_arr)
        if len(self.binary_model_.classes_) == 2:
            elevated_idx = list(self.binary_model_.classes_).index(1)
            danger_proba_elevated = danger_proba[:, elevated_idx]
        else:
            danger_proba_elevated = danger_proba[:, 0]

        mc_proba = self.multiclass_model_.predict_proba(X_arr)
        all_classes = self.multiclass_model_.classes_
        distribution = np.zeros((len(X_arr), 5))
        for i, cls in enumerate(all_classes):
            if 1 <= cls <= 5:
                distribution[:, cls - 1] = mc_proba[:, i]
        row_sums = distribution.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        distribution = distribution / row_sums

        danger_3class = self.threeclass_model_.predict(X_arr)

        return {
            "danger_binary": danger_binary,
            "danger_proba": danger_proba_elevated,
            "danger_distribution": distribution,
            "danger_3class": danger_3class,
        }

    def shap_values(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        X_arr = X.values.astype(np.float64)
        np.nan_to_num(X_arr, copy=False, nan=0.0)

        result: dict[str, np.ndarray] = {}

        explainer = shap.TreeExplainer(self.binary_model_)
        sv = explainer.shap_values(X_arr)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
        result["binary"] = sv

        mc_explainer = shap.TreeExplainer(self.multiclass_model_)
        mc_sv = mc_explainer.shap_values(X_arr)
        if isinstance(mc_sv, list):
            result["multiclass"] = mc_sv
        elif isinstance(mc_sv, np.ndarray) and mc_sv.ndim == 3:
            result["multiclass"] = [mc_sv[:, :, i] for i in range(mc_sv.shape[2])]
        else:
            result["multiclass"] = [mc_sv]

        return result

    def global_feature_importance(self) -> list[tuple[str, float]]:
        importances = self.binary_model_.feature_importances_
        pairs = list(zip(self.feature_names_, importances.tolist()))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "binary": self.binary_model_,
                "multiclass": self.multiclass_model_,
                "threeclass": self.threeclass_model_,
            },
            path / "models.joblib",
        )
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
    def load(cls, path: Path) -> Stage2RF:
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
        models = joblib.load(path / "models.joblib")
        instance.binary_model_ = models["binary"]
        instance.multiclass_model_ = models["multiclass"]
        instance.threeclass_model_ = models["threeclass"]
        instance.feature_names_ = meta["feature_names"]
        instance.smote_applied_ = meta["smote_applied"]
        instance.metadata_ = meta["metadata"]
        return instance


class TwoStageRFPipeline:
    def __init__(
        self,
        stage1_kwargs: dict | None = None,
        stage2_kwargs: dict | None = None,
    ):
        self.stage1 = Stage1RF(**(stage1_kwargs or {}))
        self.stage2 = Stage2RF(**(stage2_kwargs or {}))

    def train(
        self,
        X: pd.DataFrame,
        y_problem_types: pd.DataFrame,
        y_danger: pd.Series | np.ndarray,
    ) -> None:
        self.stage1.train(X, y_problem_types)

        stage1_probs = self.stage1.predict(X)
        stage1_probs.columns = PROB_COLUMNS
        X_stage2 = pd.concat([X, stage1_probs], axis=1)

        self.stage2.train(X_stage2, y_danger)

    def predict(self, X: pd.DataFrame) -> dict:
        stage1_probs = self.stage1.predict(X)
        stage1_probs_renamed = stage1_probs.copy()
        stage1_probs_renamed.columns = PROB_COLUMNS
        X_stage2 = pd.concat(
            [X.reset_index(drop=True), stage1_probs_renamed.reset_index(drop=True)],
            axis=1,
        )

        stage2_result = self.stage2.predict(X_stage2)

        shap_vals = self.stage2.shap_values(X_stage2)

        return {
            "danger_binary": stage2_result["danger_binary"],
            "danger_proba": stage2_result["danger_proba"],
            "danger_distribution": stage2_result["danger_distribution"],
            "danger_3class": stage2_result["danger_3class"],
            "problem_types": stage1_probs,
            "shap_values": shap_vals,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.stage1.save(path / "stage1")
        self.stage2.save(path / "stage2")

    @classmethod
    def load(cls, path: Path) -> TwoStageRFPipeline:
        path = Path(path)
        instance = cls()
        instance.stage1 = Stage1RF.load(path / "stage1")
        instance.stage2 = Stage2RF.load(path / "stage2")
        return instance
