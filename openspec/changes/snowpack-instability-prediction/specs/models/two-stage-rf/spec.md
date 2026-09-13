## Purpose

Train and evaluate a two-stage Random Forest classifier that first predicts avalanche problem types, then uses those predictions with weather features to predict danger level, with SHAP-based explanations.

## ADDED Requirements

### Requirement: Stage 1 — Problem type prediction
The system SHALL train a multi-label Random Forest classifier that predicts which avalanche problem types are active for a given zone, elevation band, and date.

#### Scenario: Stage 1 training
- **WHEN** Stage 1 training is invoked with feature data and labels
- **THEN** the system SHALL train binary Random Forest classifiers for 3 focused problem types (Persistent Slab, Slab Problem [storm+wind merged due to limited wind data], Loose Wet) per elevation band, yielding 9 Stage-1 models (3 types x 3 bands), using weather window features and physics proxy features as inputs

#### Scenario: Stage 1 prediction output
- **WHEN** Stage 1 produces predictions for a sample
- **THEN** the output SHALL include a probability score (0.0-1.0) for each of the 3 problem types via predict_proba, with a default classification threshold of 0.30 (tunable at serving time for safety)

#### Scenario: Stage 1 per-elevation-band models
- **WHEN** Stage 1 training is invoked
- **THEN** the system SHALL train separate models for each elevation band (above_treeline, near_treeline, below_treeline), yielding 9 Stage-1 models total, as problem type distributions differ significantly by elevation

### Requirement: Stage 2 — Danger level prediction
The system SHALL train a Random Forest classifier that predicts the 1-5 danger level using weather features, physics proxies, AND the Stage 1 problem type predictions as additional input features.

#### Scenario: Frozen Stage 1 predictions for Stage 2
- **WHEN** Stage 2 training data is prepared
- **THEN** the system SHALL use out-of-sample Stage-1 ensemble predictions (never actual labels) as Stage 2 input features, generated via a staged chronological split protocol to prevent label leakage

#### Scenario: Stage 2 training
- **WHEN** Stage 2 training is invoked
- **THEN** the system SHALL train a Random Forest using the concatenation of (weather window features, physics proxy features, frozen Stage 1 predicted problem type probabilities) as the input feature vector, with the ordinal danger level (1-5) as the target, yielding 3 Stage-2 models (1 per elevation band) for 12 total models

#### Scenario: Stage 2 prediction output
- **WHEN** Stage 2 produces predictions for a sample
- **THEN** the output SHALL include the predicted danger level (integer 1-5) and a probability distribution over all 5 levels

#### Scenario: Stage 2 per-elevation-band models
- **WHEN** Stage 2 training is invoked
- **THEN** the system SHALL train separate models for each elevation band, consistent with Stage 1

### Requirement: Class imbalance handling
The system SHALL address severe class imbalance (approximately 1.1% of days at danger level 4-5) using cost-sensitive learning only. SMOTE is explicitly excluded — research confirmed it hurts generalization.

#### Scenario: Cost-sensitive learning
- **WHEN** Random Forest models are trained
- **THEN** the system SHALL use cost-sensitive class weights ({0:1, 1:5} baseline) to penalize misclassification of rare classes, without SMOTE oversampling

#### Scenario: Threshold tuning at serving time
- **WHEN** models are deployed for prediction
- **THEN** the system SHALL deploy with predict_proba and a configurable classification threshold (default t=0.30 for safety), adjustable per deployment context without retraining

### Requirement: Soft-voting ensemble
The system SHALL combine multiple RF configurations via soft-voting to improve robustness.

#### Scenario: Ensemble construction
- **WHEN** hyperparameter search completes
- **THEN** the system SHALL select the top-3 configurations ranked by macro-F1 on the validation set and combine them via soft-voting (averaged probability distributions) for each stage and elevation band

### Requirement: SHAP explanations
The system SHALL provide feature-level explanations for each prediction using SHAP (SHapley Additive exPlanations).

#### Scenario: Per-prediction SHAP values
- **WHEN** a prediction is generated
- **THEN** the system SHALL compute SHAP values for all input features and return the top 10 features by absolute SHAP value, with their direction (positive/negative contribution to the predicted class)

#### Scenario: SHAP for Stage 2 including problem type features
- **WHEN** SHAP values are computed for a Stage 2 prediction
- **THEN** the explanation SHALL include SHAP values for the Stage 1 problem type probability features alongside weather and physics proxy features, enabling end-to-end interpretability

### Requirement: Temporal train/validation/test split
The system SHALL use strictly temporal splits to prevent data leakage from temporal autocorrelation.

#### Scenario: Default split boundaries
- **WHEN** training is invoked without explicit split dates
- **THEN** the system SHALL use training: 2013-12-01 to 2022-06-30, validation: 2022-10-01 to 2024-06-30 (2022-24 seasons), test: 2024-10-01 to 2025-06-30 (2024-25 season)

#### Scenario: No temporal leakage
- **WHEN** data is split for training
- **THEN** no observation from a date in the validation or test period SHALL appear in the training set, including within rolling window features that look backward from the split boundary

### Requirement: Evaluation metrics
The system SHALL evaluate model performance using metrics appropriate for ordinal and imbalanced classification.

#### Scenario: Metric computation on test set
- **WHEN** model evaluation is performed
- **THEN** the system SHALL report: macro-averaged F1, per-class F1, ordinal accuracy (within ±1 of true label), confusion matrix, and ROC-AUC per class

#### Scenario: Per-elevation-band metrics
- **WHEN** evaluation is performed
- **THEN** the system SHALL report all metrics separately for each elevation band in addition to the aggregate

#### Scenario: Benchmark comparison
- **WHEN** evaluation is performed
- **THEN** the system SHALL compare results against: (1) a climatological baseline (predicting the most frequent class per season/elevation), and (2) the published Schwartzreich & Rodriguez 2026 metrics where available

### Requirement: Experiment tracking
The system SHALL log all training runs to MLflow with sufficient detail for reproducibility.

#### Scenario: MLflow run logging
- **WHEN** a training run completes
- **THEN** the system SHALL log to MLflow: hyperparameters, feature list, split dates, all evaluation metrics, the trained model artifact, and the SHAP summary plot as an artifact
