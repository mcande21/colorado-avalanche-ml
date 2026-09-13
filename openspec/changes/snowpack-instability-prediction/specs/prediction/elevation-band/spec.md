## Purpose

Produce avalanche danger predictions segmented by elevation band with both categorical and probabilistic outputs, providing actionable forecasts for above, near, and below treeline zones.

## ADDED Requirements

### Requirement: Elevation band definitions
The system SHALL define and apply consistent elevation band boundaries for prediction.

#### Scenario: Standard elevation bands
- **WHEN** predictions are requested for a zone
- **THEN** the system SHALL produce separate predictions for three elevation bands: above_treeline (approximately > 11,500 ft in Colorado, zone-dependent), near_treeline (approximately 10,500-11,500 ft), and below_treeline (approximately < 10,500 ft)

#### Scenario: Zone-specific treeline calibration
- **WHEN** elevation band boundaries are determined for a CAIC zone
- **THEN** the system SHALL use the treeline elevation specific to that zone's latitude and aspect distribution, as provided by zone metadata, rather than a single statewide threshold

### Requirement: Prediction output format
The system SHALL produce a structured prediction output for each zone-elevation-band-date combination.

#### Scenario: Complete prediction output
- **WHEN** a prediction is generated for a zone, elevation band, and date
- **THEN** the output SHALL include: predicted danger level (integer 1-5), probability distribution over all 5 danger levels (summing to 1.0), predicted problem types (list of types with probability > threshold), and the model identifier (rf or lstm) used

#### Scenario: Confidence indication
- **WHEN** a prediction is generated
- **THEN** the output SHALL include a confidence score computed as the maximum class probability minus the second-highest class probability, providing a margin-of-victory metric

### Requirement: Problem type association with elevation bands
The system SHALL associate predicted problem types with the elevation bands where they are most likely.

#### Scenario: Problem type per elevation band
- **WHEN** problem type predictions are generated
- **THEN** each predicted problem type SHALL be reported per elevation band, reflecting that different problem types dominate at different elevations (e.g., Wind Slab predominantly above treeline, Loose Wet predominantly below treeline)

### Requirement: Multi-model ensemble support
The system SHALL support combining predictions from the RF and LSTM models.

#### Scenario: Ensemble prediction
- **WHEN** both RF and LSTM models have been trained and an ensemble prediction is requested
- **THEN** the system SHALL produce an ensemble prediction by averaging the probability distributions from both models, weighted by their validation set performance (inverse validation loss weighting)

#### Scenario: Single model fallback
- **WHEN** only one model is available (e.g., LSTM not yet trained)
- **THEN** the system SHALL use the available model's prediction directly without ensemble weighting

### Requirement: Prediction consistency constraints
The system SHALL enforce physical consistency constraints across elevation band predictions.

#### Scenario: Monotonic danger constraint
- **WHEN** predictions are generated for all three elevation bands of a single zone and date
- **THEN** the system SHALL flag (but not override) cases where the above_treeline danger is lower than near_treeline, or near_treeline is lower than below_treeline, as these are unusual but not physically impossible

#### Scenario: Problem type coherence
- **WHEN** problem types are predicted across elevation bands
- **THEN** the system SHALL flag cases where physically elevation-dependent problem types appear at unexpected bands (e.g., Cornice Fall below treeline) as low-confidence predictions

### Requirement: Historical prediction storage
The system SHALL store all predictions for historical analysis and model monitoring.

#### Scenario: Prediction persistence
- **WHEN** predictions are generated
- **THEN** the system SHALL store each prediction in DuckDB with columns (zone_id, date, elevation_band, model_id, predicted_danger, danger_probabilities, predicted_problem_types, confidence, generated_at), enabling retrospective analysis and model drift detection

#### Scenario: Prediction vs actual tracking
- **WHEN** actual CAIC danger ratings become available for a date that has predictions
- **THEN** the system SHALL compute and store the prediction error (predicted - actual) for each zone-elevation-band, enabling ongoing accuracy monitoring
