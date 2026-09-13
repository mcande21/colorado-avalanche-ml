## Purpose

Train and evaluate a hierarchical multi-rate LSTM with N-HiTS-style temporal pooling that captures patterns across hourly, daily, and seasonal time scales for avalanche danger prediction.

## ADDED Requirements

### Requirement: Multi-rate temporal architecture
The system SHALL implement a hierarchical LSTM with three temporal branches operating at different resolutions, inspired by N-HiTS multi-rate MaxPool architecture.

#### Scenario: Branch 1 — Hourly resolution
- **WHEN** the model processes an input sample
- **THEN** Branch 1 SHALL consume the most recent 7 days of hourly observations (168 time steps), applying MaxPool to reduce temporal resolution before LSTM encoding

#### Scenario: Branch 2 — Daily resolution
- **WHEN** the model processes an input sample
- **THEN** Branch 2 SHALL consume the most recent 30 days of daily aggregated observations (30 time steps) processed through a separate LSTM encoder

#### Scenario: Branch 3 — Seasonal resolution
- **WHEN** the model processes an input sample
- **THEN** Branch 3 SHALL consume the full season history using 3-day aggregated windows (up to 60 time steps for a 180-day season) processed through a separate LSTM encoder

#### Scenario: Branch concatenation
- **WHEN** all three branches produce their hidden state outputs
- **THEN** the system SHALL concatenate the final hidden states of all three branches and pass the concatenated vector through dense layers to produce the prediction

### Requirement: Two-stage prediction structure
The system SHALL implement the same two-stage prediction structure as the Random Forest model: Stage 1 for problem types, Stage 2 for danger level.

#### Scenario: Stage 1 problem type head
- **WHEN** the concatenated branch output is produced
- **THEN** a multi-label classification head SHALL predict probabilities for each of the 9 problem types using sigmoid activation per output

#### Scenario: Stage 2 danger level head
- **WHEN** Stage 1 predictions are produced
- **THEN** a second classification head SHALL take the concatenated branch output PLUS Stage 1 predicted probabilities as input and produce a probability distribution over danger levels 1-5 via softmax

#### Scenario: Joint training
- **WHEN** the model is trained
- **THEN** both stages SHALL be trained jointly using a weighted sum of Stage 1 multi-label binary cross-entropy loss and Stage 2 ordinal cross-entropy loss

### Requirement: Input feature handling
The system SHALL handle the input feature requirements for each temporal branch.

#### Scenario: Hourly branch input features
- **WHEN** hourly data is prepared for Branch 1
- **THEN** input features SHALL include: air_temp, precip, wind_speed, wind_direction, swe, snow_depth, and HRRR-interpolated fields where available, normalized per-station using training set statistics

#### Scenario: Daily branch input features
- **WHEN** daily data is prepared for Branch 2
- **THEN** input features SHALL include: daily aggregated weather statistics (min/max/mean temp, total precip, max wind), physics proxy features (temp gradient, surface hoar index, wind slab loading, rain-on-snow), and SWE/snow depth change

#### Scenario: Seasonal branch input features
- **WHEN** seasonal data is prepared for Branch 3
- **THEN** input features SHALL include: 3-day aggregated weather summaries, cumulative season metrics (total SWE, cumulative precip, faceting days), and snow depth anomaly

#### Scenario: Missing data masking
- **WHEN** input sequences contain missing values
- **THEN** the system SHALL apply a binary mask indicating valid observations, and the LSTM SHALL use masked attention or zero-filling with the mask concatenated as an additional input feature

### Requirement: Transformer option for Persistent Slab
The system SHALL support a Transformer architecture as an alternative Stage-1 head for the Persistent Slab problem type, based on research showing Transformer superiority for this specific prediction target.

#### Scenario: Transformer Persistent Slab head
- **WHEN** the persistent slab Stage-1 model is configured
- **THEN** the system SHALL support a Transformer head option (64-unit, 2-layer) as an alternative to LSTM, selected by validation performance comparison

#### Scenario: Transformer benchmark
- **WHEN** the Transformer head is evaluated
- **THEN** the system SHALL compare against LSTM and RF for the Persistent Slab Stage-1 model specifically, reporting macro-F1 per elevation band, as Schwartzreich 2026 found Transformer outperformed RF, LSTM, and GRU for this problem type

### Requirement: Per-elevation-band prediction
The system SHALL produce separate predictions per elevation band using the same 12-model structure as the RF.

#### Scenario: Elevation band handling
- **WHEN** the model is configured
- **THEN** the system SHALL train 9 Stage-1 models (3 problem types x 3 elevation bands) + 3 Stage-2 models (1 per band) = 12 total, matching the RF architecture structure

### Requirement: Class imbalance handling
The system SHALL address class imbalance in the deep learning context.

#### Scenario: Focal loss for rare classes
- **WHEN** the model is trained
- **THEN** the system SHALL use focal loss (gamma configurable, default 2.0) for the danger level prediction head to down-weight easy examples and focus on rare high-danger events

#### Scenario: Balanced sampling
- **WHEN** training batches are constructed
- **THEN** the system SHALL use a weighted random sampler that oversamples minority classes to achieve approximately equal representation per batch

### Requirement: Evaluation and benchmarking
The system SHALL be evaluated against the Random Forest baseline using identical data splits and metrics.

#### Scenario: Same test protocol as RF
- **WHEN** the hierarchical LSTM is evaluated
- **THEN** the system SHALL use the identical temporal train/validation/test split as the two-stage RF and report the same metrics: macro-averaged F1, per-class F1, ordinal accuracy, confusion matrix, ROC-AUC per class, all per elevation band

#### Scenario: RF benchmark comparison
- **WHEN** evaluation metrics are computed
- **THEN** the system SHALL report a direct comparison table between the hierarchical LSTM and two-stage RF, with statistical significance testing (paired bootstrap, p < 0.05) on the test set

#### Scenario: Published benchmark comparison
- **WHEN** evaluation is complete
- **THEN** the system SHALL compare against Schwartzreich & Rodriguez 2026 published metrics where available and note any differences in data coverage, label source, or evaluation protocol

### Requirement: Experiment tracking
The system SHALL log all training runs with full reproducibility information.

#### Scenario: MLflow run logging
- **WHEN** a training run completes
- **THEN** the system SHALL log to MLflow: architecture configuration (hidden sizes, dropout, learning rate, batch size, number of epochs), training curves (loss per epoch), all evaluation metrics, model checkpoint artifact, and comparison metrics against the RF baseline
