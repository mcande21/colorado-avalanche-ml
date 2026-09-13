## Purpose

Ingest HRRR 3km gridded weather forecast data from AWS S3 via Herbie, applying SNOTEL-based precipitation bias correction before storage.

## ADDED Requirements

### Requirement: HRRR grid retrieval
The system SHALL retrieve HRRR analysis and forecast grids from the AWS HRRR archive via the Herbie package, extracting variables relevant to avalanche prediction.

#### Scenario: Successful grid retrieval for a date
- **WHEN** HRRR ingestion is invoked for a date and forecast hour
- **THEN** the system SHALL retrieve HRRR GRIB2 data from AWS S3 via Herbie, extract variables (2m temperature, 10m wind speed, 10m wind direction, total precipitation, snow water equivalent, relative humidity, surface pressure, downward shortwave radiation), and store gridpoint values for the Colorado mountain domain

#### Scenario: Required spatial domain
- **WHEN** HRRR grids are retrieved
- **THEN** the system SHALL extract gridpoints covering the Colorado mountain zones (approximately 37°N-41°N, 105°W-108°W) at the native 3km resolution

#### Scenario: HRRR archive gap
- **WHEN** HRRR data is unavailable for a requested date (archive gap or AWS outage)
- **THEN** the system SHALL log the missing date, skip it, and continue processing remaining dates without halting the pipeline

### Requirement: Precipitation bias correction
The system SHALL apply bias correction to HRRR precipitation fields using co-located SNOTEL observations to address the documented 25-65% precipitation bias.

#### Scenario: Station-based bias correction
- **WHEN** HRRR precipitation values are extracted for a gridpoint near a SNOTEL station
- **THEN** the system SHALL compute a correction factor as (SNOTEL_observed / HRRR_predicted) using a rolling 30-day window, and apply the factor to the HRRR precipitation field

#### Scenario: Gridpoint with no nearby station
- **WHEN** a HRRR gridpoint has no SNOTEL station within 15km
- **THEN** the system SHALL apply the inverse-distance-weighted average correction factor from the 3 nearest stations

#### Scenario: Zero precipitation in HRRR
- **WHEN** HRRR predicts zero precipitation but SNOTEL observes non-zero precipitation
- **THEN** the system SHALL flag the gridpoint-date as a potential missed precipitation event and apply the regional mean correction factor

### Requirement: HRRR-to-station interpolation
The system SHALL interpolate HRRR gridded values to SNOTEL station locations for feature enrichment.

#### Scenario: Bilinear interpolation to station
- **WHEN** HRRR data is requested for a specific SNOTEL station location
- **THEN** the system SHALL perform bilinear interpolation from the surrounding 4 HRRR gridpoints to the station coordinates, with elevation lapse-rate adjustment for temperature (6.5°C/km standard lapse rate)

### Requirement: Temporal resolution handling
The system SHALL support both HRRR analysis (hour 0) and short-range forecast fields.

#### Scenario: Analysis field ingestion
- **WHEN** analysis fields (f00) are requested
- **THEN** the system SHALL retrieve the HRRR analysis valid at each hour, providing hourly observed-equivalent gridded data

#### Scenario: Forecast field ingestion
- **WHEN** forecast fields are requested for lead times beyond hour 0
- **THEN** the system SHALL retrieve the specified forecast hours (up to f18) and tag stored values with both the initialization time and the valid time

### Requirement: Storage format
The system SHALL store bias-corrected HRRR data in DuckDB, organized for efficient station-level and gridpoint-level queries.

#### Scenario: Station-interpolated storage
- **WHEN** HRRR values interpolated to station locations are stored
- **THEN** the system SHALL store them with columns (station_id, valid_time, init_time, temp_2m_k, wind_speed_10m_ms, wind_dir_10m_deg, precip_mm, swe_mm, rh_pct, pressure_hpa, sw_rad_wm2, precip_bias_corrected) in a station-indexed table

#### Scenario: Gridded storage for spatial downscaling
- **WHEN** full HRRR grids are stored for spatial downscaling use
- **THEN** the system SHALL store them with columns (lat, lon, valid_time, variable, value) in a gridpoint-indexed table partitioned by date
