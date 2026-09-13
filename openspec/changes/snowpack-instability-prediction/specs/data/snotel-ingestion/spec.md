## Purpose

Ingest and store SNOTEL station observations from the AWDB REST API for all 118 Colorado stations, supporting both hourly and daily temporal resolutions in DuckDB.

## ADDED Requirements

### Requirement: Station discovery and metadata persistence
The system SHALL discover all active SNOTEL stations in Colorado via the AWDB REST API and persist station metadata including station ID, name, latitude, longitude, elevation, and HUC boundary.

#### Scenario: Initial station catalog load
- **WHEN** the ingestion system is initialized with no prior station metadata
- **THEN** the system SHALL query the AWDB API for all CO SNOTEL stations, store metadata for all active stations (expected ~118), and record the catalog retrieval timestamp

#### Scenario: Station metadata refresh
- **WHEN** the station catalog is older than 30 days
- **THEN** the system SHALL re-query the AWDB API, add any new stations, mark decommissioned stations as inactive, and update changed metadata fields

### Requirement: Hourly observation ingestion
The system SHALL ingest hourly observations from SNOTEL stations including snow water equivalent (SWE), snow depth, air temperature, and precipitation accumulation.

#### Scenario: Successful hourly ingestion for a date range
- **WHEN** the ingestion process is invoked for a station and date range
- **THEN** the system SHALL retrieve hourly records from the AWDB API, store them in DuckDB with columns (station_id, timestamp, swe_inches, snow_depth_inches, air_temp_f, precip_accum_inches), and record the ingestion watermark

#### Scenario: Partial data availability
- **WHEN** a station reports some but not all variables for a given hour
- **THEN** the system SHALL store the available values and set missing values to NULL, without skipping the record

#### Scenario: Station offline or no data returned
- **WHEN** the AWDB API returns no data for a station and date range
- **THEN** the system SHALL log a warning with station ID and date range, skip that station, and continue ingesting remaining stations

### Requirement: Daily observation ingestion
The system SHALL ingest daily summary observations including daily SWE, snow depth, min/max/mean air temperature, and precipitation increment.

#### Scenario: Successful daily ingestion
- **WHEN** the daily ingestion process is invoked for a station and date range
- **THEN** the system SHALL retrieve daily records, store them with columns (station_id, date, swe_inches, snow_depth_inches, air_temp_min_f, air_temp_max_f, air_temp_mean_f, precip_increment_inches), and update the ingestion watermark

#### Scenario: Duplicate ingestion for overlapping date range
- **WHEN** ingestion is invoked for a date range that overlaps previously ingested data
- **THEN** the system SHALL upsert records keyed on (station_id, date), replacing existing values with the newly retrieved values

### Requirement: Incremental ingestion
The system SHALL support incremental ingestion by tracking a per-station, per-resolution watermark indicating the last successfully ingested timestamp.

#### Scenario: Incremental daily run
- **WHEN** the daily ingestion scheduler triggers
- **THEN** the system SHALL query each station's watermark, request only data after the watermark from the AWDB API, and advance the watermark on successful storage

#### Scenario: Backfill for a specific date range
- **WHEN** a backfill is requested with explicit start and end dates
- **THEN** the system SHALL ignore the watermark, ingest the full requested range, and update the watermark to the latest date if it exceeds the current watermark

### Requirement: Data quality validation
The system SHALL validate ingested observations against physical plausibility bounds before storage.

#### Scenario: Value outside physical bounds
- **WHEN** an observation contains SWE < 0, snow depth < 0, or air temperature outside [-60°F, 120°F]
- **THEN** the system SHALL flag the record with a quality flag, store it with the flag, and exclude flagged records from downstream feature computation by default

#### Scenario: Sudden discontinuity detection
- **WHEN** SWE or snow depth changes by more than 200% relative to the prior reading within a single time step
- **THEN** the system SHALL flag the record as a potential sensor error

### Requirement: AWDB API rate limiting and retry
The system SHALL respect AWDB API rate limits and implement retry with exponential backoff on transient failures.

#### Scenario: Transient API failure
- **WHEN** the AWDB API returns a 5xx error or connection timeout
- **THEN** the system SHALL retry up to 3 times with exponential backoff (1s, 2s, 4s base delays) before marking the station-date range as failed

#### Scenario: Rate limit exceeded
- **WHEN** the AWDB API returns a 429 response
- **THEN** the system SHALL pause ingestion for the duration indicated by the Retry-After header, or 60 seconds if no header is present, before resuming
