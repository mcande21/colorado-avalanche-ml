## Purpose

Serve avalanche danger predictions and supporting data via a FastAPI service with a daily ingestion scheduler, enabling programmatic access to forecasts.

## ADDED Requirements

### Requirement: Prediction endpoint
The system SHALL expose a POST /predict endpoint that returns an avalanche danger prediction for a specified zone, elevation band, and date.

#### Scenario: Successful prediction request
- **WHEN** a POST request is sent to /predict with body {"zone_id": "<zone>", "elevation_band": "<band>", "date": "<date>"}
- **THEN** the response SHALL return HTTP 200 with a JSON body containing: danger_level (int 1-5), danger_probabilities (array of 5 floats summing to 1.0), problem_types (array of objects with type and probability), shap_explanations (array of top 10 features with name, value, and contribution), confidence (float), and model_id (string)

#### Scenario: Prediction for current day
- **WHEN** a POST request is sent to /predict with date equal to today
- **THEN** the system SHALL use the most recently ingested feature data to generate a real-time prediction

#### Scenario: Prediction for future date
- **WHEN** a POST request is sent to /predict with a date more than 1 day in the future
- **THEN** the response SHALL return HTTP 400 with an error message indicating that predictions beyond T+1 are not supported without HRRR forecast data

#### Scenario: Invalid zone or elevation band
- **WHEN** a POST request is sent with a zone_id or elevation_band not in the system's supported set
- **THEN** the response SHALL return HTTP 422 with an error message listing the valid zone IDs or elevation bands

#### Scenario: Insufficient feature data
- **WHEN** the feature store lacks sufficient data for the requested zone and date (e.g., station offline for > 120h)
- **THEN** the response SHALL return HTTP 200 with the prediction plus a warnings array containing "limited_data" and the coverage fraction

### Requirement: History endpoint
The system SHALL expose a GET /history endpoint that returns historical predictions and actuals for a zone.

#### Scenario: Successful history request
- **WHEN** a GET request is sent to /history with query parameters zone_id, elevation_band, start_date, and end_date
- **THEN** the response SHALL return HTTP 200 with an array of daily records, each containing: date, predicted_danger, actual_danger (if available), danger_probabilities, predicted_problem_types, and prediction_error (if actual is available)

#### Scenario: Date range limits
- **WHEN** a GET request to /history spans more than 365 days
- **THEN** the response SHALL return HTTP 400 indicating maximum range is 365 days

### Requirement: Stations endpoint
The system SHALL expose a GET /stations endpoint listing available SNOTEL stations and their status.

#### Scenario: Station listing
- **WHEN** a GET request is sent to /stations
- **THEN** the response SHALL return HTTP 200 with an array of station objects containing: station_id, name, latitude, longitude, elevation_ft, associated_zones (array of zone IDs), last_observation_date, and status (active/inactive)

#### Scenario: Station filtering
- **WHEN** a GET request is sent to /stations with query parameter zone_id
- **THEN** the response SHALL return only stations associated with the specified zone

### Requirement: Health endpoint
The system SHALL expose a GET /health endpoint for monitoring service availability and data freshness.

#### Scenario: Healthy service
- **WHEN** a GET request is sent to /health and all subsystems are operational
- **THEN** the response SHALL return HTTP 200 with: status "healthy", last_ingestion_time, feature_store_row_count, model_version, and data_freshness_hours (hours since last ingestion completed)

#### Scenario: Stale data warning
- **WHEN** a GET request is sent to /health and the last successful ingestion was more than 36 hours ago
- **THEN** the response SHALL return HTTP 200 with status "degraded" and a warnings array containing "stale_data" with the hours since last ingestion

#### Scenario: Database unreachable
- **WHEN** a GET request is sent to /health and DuckDB is unreachable
- **THEN** the response SHALL return HTTP 503 with status "unhealthy" and error details

### Requirement: Daily ingestion scheduler
The system SHALL run an automated daily data ingestion and feature computation pipeline.

#### Scenario: Scheduled daily run
- **WHEN** the scheduled ingestion time is reached (configurable, default 06:00 UTC)
- **THEN** the system SHALL execute in order: (1) SNOTEL observation ingestion for the past 48h, (2) CAIC label ingestion for the past 48h, (3) HRRR ingestion for the past 48h (if enabled), (4) feature computation for affected dates, (5) prediction generation for today and tomorrow (if HRRR forecast available)

#### Scenario: Ingestion failure isolation
- **WHEN** one stage of the daily pipeline fails
- **THEN** the system SHALL log the failure, continue with remaining stages that do not depend on the failed stage, and report the failure status via the /health endpoint

#### Scenario: Manual ingestion trigger
- **WHEN** a POST request is sent to /ingest with an API key
- **THEN** the system SHALL trigger the full ingestion pipeline immediately, returning HTTP 202 with a job ID for status polling

### Requirement: Authentication and rate limiting
The system SHALL protect endpoints with API key authentication and rate limiting.

#### Scenario: Missing API key
- **WHEN** a request is sent to any endpoint except /health without an X-API-Key header
- **THEN** the response SHALL return HTTP 401 with error "API key required"

#### Scenario: Rate limit exceeded
- **WHEN** a client exceeds 100 requests per minute
- **THEN** the response SHALL return HTTP 429 with a Retry-After header

### Requirement: Docker deployment
The system SHALL be deployable as a Docker container with configurable environment.

#### Scenario: Container startup
- **WHEN** the Docker container starts with required environment variables (DUCKDB_PATH, MODEL_PATH, API_KEY)
- **THEN** the service SHALL load the model, verify DuckDB connectivity, and begin accepting requests within 30 seconds

#### Scenario: Missing required configuration
- **WHEN** the container starts without a required environment variable
- **THEN** the container SHALL exit with a non-zero code and a log message identifying the missing variable

### Requirement: Request/response validation
The system SHALL validate all request inputs and produce consistent error responses.

#### Scenario: Malformed request body
- **WHEN** a POST request contains invalid JSON or missing required fields
- **THEN** the response SHALL return HTTP 422 with a structured error listing each validation failure

#### Scenario: Consistent error format
- **WHEN** any error response is returned
- **THEN** the body SHALL follow the format {"error": "<code>", "message": "<human-readable>", "details": [<optional array>]}
