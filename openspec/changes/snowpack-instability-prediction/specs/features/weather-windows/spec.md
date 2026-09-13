## Purpose

Compute rolling window aggregations over weather time series at multiple temporal scales, producing standardized feature vectors for model consumption.

## ADDED Requirements

### Requirement: Rolling window aggregation
The system SHALL compute rolling window features over weather observations at windows of 24h, 48h, 72h, 96h, and 120h for each SNOTEL station.

#### Scenario: Temperature window features
- **WHEN** features are computed for a station and date
- **THEN** the system SHALL produce for each window size: temp_mean, temp_min, temp_max, temp_std, temp_range (max-min), and temp_trend (linear slope over window)

#### Scenario: Precipitation window features
- **WHEN** features are computed for a station and date
- **THEN** the system SHALL produce for each window size: precip_sum (total accumulation), precip_max_hourly (peak hourly rate), precip_hours (count of hours with precip > 0), and precip_intensity (sum / precip_hours)

#### Scenario: Wind window features
- **WHEN** features are computed for a station and date
- **THEN** the system SHALL produce for each window size: wind_mean, wind_max, wind_std, and wind_direction_mode (circular mode of wind direction)

#### Scenario: SWE and snow depth window features
- **WHEN** features are computed for a station and date
- **THEN** the system SHALL produce for each window size: swe_change (delta over window), snow_depth_change, swe_rate (change per hour), and new_snow (positive snow depth change, clamped to zero if negative)

### Requirement: Window computation with missing data
The system SHALL handle gaps in the observation time series during window computation.

#### Scenario: Partial data within window
- **WHEN** fewer than 50% of expected hourly observations are present within a window
- **THEN** the system SHALL mark all features for that window as NULL rather than computing from sparse data

#### Scenario: Minor gaps within window
- **WHEN** 50% or more of expected hourly observations are present within a window
- **THEN** the system SHALL compute features from available data and include a coverage_fraction column indicating the proportion of non-null observations

### Requirement: Feature vector output schema
The system SHALL produce a consistent feature vector schema across all stations and dates.

#### Scenario: Feature vector for a station-date
- **WHEN** all window features are computed for a station and date
- **THEN** the system SHALL produce a single row in DuckDB with columns named as {variable}_{aggregation}_{window}h (e.g., temp_mean_24h, precip_sum_72h, wind_max_120h), plus metadata columns (station_id, date, coverage_fraction_per_window)

#### Scenario: Feature count
- **WHEN** the full feature vector is assembled from weather windows alone
- **THEN** it SHALL contain exactly (4 temp stats + temp_range + temp_trend) * 5 windows + (4 precip stats) * 5 windows + (4 wind stats) * 5 windows + (4 SWE/snow stats) * 5 windows = 90 weather window features per station-date

### Requirement: Incremental feature computation
The system SHALL support incremental feature computation when new observations arrive.

#### Scenario: Daily feature refresh
- **WHEN** new observations are ingested for a station
- **THEN** the system SHALL recompute features only for dates affected by the new data (dates within 120h of any new observation) and upsert the results into the feature store
