## Purpose

Spatially downscale HRRR 3km gridded weather data to arbitrary mountain locations using DEM terrain features and SNOTEL bias correction, enabling prediction at locations without weather stations.

## ADDED Requirements

### Requirement: DEM terrain feature extraction
The system SHALL extract terrain features from the USGS 3DEP digital elevation model for any location in the Colorado mountain domain.

#### Scenario: Terrain feature computation for a point
- **WHEN** terrain features are requested for a latitude/longitude coordinate
- **THEN** the system SHALL compute: elevation (m), slope (degrees), aspect (degrees from north), topographic position index at 500m radius (TPI@500m), sky view factor (SVF), and profile/plan curvature from the 3DEP DEM

#### Scenario: DEM resolution
- **WHEN** terrain features are computed
- **THEN** the system SHALL use at minimum 10m resolution 3DEP data, resampled from the 1/3 arc-second (~10m) product

#### Scenario: Point outside Colorado mountain domain
- **WHEN** terrain features are requested for a location outside the supported domain (37°N-41°N, 105°W-108°W)
- **THEN** the system SHALL return an error indicating the location is outside the supported domain

### Requirement: XGBoost downscaling model
The system SHALL train an XGBoost model that maps (HRRR gridpoint values, terrain features) to observed SNOTEL values, learning the relationship between coarse-grid weather and station-observed weather conditioned on terrain.

#### Scenario: Downscaling model training
- **WHEN** the downscaling model is trained
- **THEN** the system SHALL use SNOTEL observations as truth, with input features: HRRR bilinearly-interpolated values at the station location, elevation difference between HRRR gridpoint and station, station terrain features (slope, aspect, TPI, SVF, curvature), and HRRR spatial gradients (difference between adjacent gridpoints)

#### Scenario: Downscaling model prediction
- **WHEN** a downscaled weather estimate is requested for an arbitrary location
- **THEN** the system SHALL extract HRRR values at the nearest gridpoints, compute terrain features for the target location, and apply the trained XGBoost model to produce corrected estimates for temperature, precipitation, wind speed, wind direction, humidity, and SWE

#### Scenario: Temporal generalization
- **WHEN** the downscaling model is evaluated
- **THEN** the system SHALL report leave-one-station-out cross-validation metrics demonstrating spatial generalization, in addition to temporal test set metrics

### Requirement: Cold air pool handling
The system SHALL account for cold air pooling in valleys and basins, where inversions cause HRRR to overpredict temperature.

#### Scenario: Cold air pool detection
- **WHEN** downscaling predictions are generated for locations with TPI@500m < -50m (valley/basin positions)
- **THEN** the system SHALL apply a cold air pool correction factor trained on SNOTEL stations in similar topographic positions, reducing temperature estimates during clear-sky nighttime conditions

#### Scenario: TPI-based correction magnitude
- **WHEN** cold air pool correction is applied
- **THEN** the correction magnitude SHALL scale with TPI magnitude and the inversion strength proxy (wind speed < 3 m/s AND clear sky indicator), with maximum correction learned from training data

### Requirement: SNOTEL bias correction integration
The system SHALL integrate the precipitation bias correction from the HRRR ingestion capability into the downscaling pipeline.

#### Scenario: Bias-corrected input to downscaling
- **WHEN** HRRR data is used as input to the downscaling model
- **THEN** precipitation fields SHALL already have the SNOTEL-based bias correction applied, as produced by the data/hrrr-ingestion capability

#### Scenario: Residual bias at non-station locations
- **WHEN** downscaled precipitation is predicted for a location distant from all SNOTEL stations
- **THEN** the system SHALL report an uncertainty estimate reflecting the extrapolation distance from the nearest bias-correction anchor stations

### Requirement: TopoPyScale baseline comparison
The system SHALL compare the XGBoost downscaling model against TopoPyScale as a physically-based baseline.

#### Scenario: TopoPyScale benchmark
- **WHEN** the downscaling model is evaluated
- **THEN** the system SHALL run TopoPyScale for the same station locations and time period, and report comparative metrics (RMSE, MAE, bias, correlation) for temperature, precipitation, and wind speed at held-out SNOTEL stations

### Requirement: Downscaled output format
The system SHALL produce downscaled weather estimates in a format compatible with the feature engineering pipeline.

#### Scenario: Feature pipeline compatibility
- **WHEN** downscaled weather values are produced for a non-station location
- **THEN** the output SHALL have the same schema as station observations (timestamp, temp, precip, wind_speed, wind_direction, swe, snow_depth, humidity), enabling the weather window and physics proxy feature pipelines to process them identically to station data

#### Scenario: Uncertainty quantification
- **WHEN** downscaled values are produced
- **THEN** each variable SHALL include a prediction interval (10th and 90th percentiles) derived from the XGBoost quantile regression or conformal prediction, in addition to the point estimate
