## 1. Project Setup

- [ ] 1.1 Create Python package structure (src/avalanche_ml/ with __init__.py, pyproject.toml with all dependencies, pytest config) and verify `pip install -e .` succeeds and `python -c "import avalanche_ml"` imports without error
- [ ] 1.2 Set up DuckDB schema creation script (stations, snotel_hourly, snotel_daily, caic_danger, caic_problems, zone_station_map, ingestion_watermarks tables) and verify tables are created by running the script and querying DuckDB for table names
- [ ] 1.3 Set up MLflow tracking configuration (local file store, experiment naming convention) and verify `mlflow.start_run()` creates a run in the configured store

## 2. SNOTEL Data Ingestion

- [ ] 2.1 Implement SnotelClient.discover() to query AWDB REST API for all CO SNOTEL stations and persist metadata to the stations table; verify by running discover and confirming ~118 stations are stored with lat/lon/elevation
- [ ] 2.2 Implement SnotelClient.ingest() for hourly observations with quality flag validation (SWE < 0, temp outside [-60F, 120F], 200% discontinuity detection); verify by ingesting 7 days for 3 test stations and confirming rows in snotel_hourly with expected columns and quality flags on injected bad values
- [ ] 2.3 Implement SnotelClient.ingest() for daily observations with upsert on (station_id, date); verify by ingesting overlapping date ranges and confirming no duplicate rows
- [ ] 2.4 Implement watermark-based incremental ingestion (ingestion_watermarks table, only fetch data after last watermark) and backfill mode (ignore watermark for explicit date ranges); verify by running incremental twice and confirming the second run requests only new data
- [ ] 2.5 Implement AWDB API retry with exponential backoff (3 retries on 5xx, pause on 429 with Retry-After); verify with a mock server returning 503 then 200 that the client retries and succeeds

## 3. CAIC Label Ingestion

- [ ] 3.1 Implement CaicClient for danger rating ingestion via caic-python, storing (zone_id, date, elevation_band, danger_level, source) in caic_danger; verify by ingesting a known date range and confirming danger levels 1-5 appear per elevation band
- [ ] 3.2 Implement avalanche problem type ingestion with normalization to the 9-type vocabulary (Persistent Slab, Storm Slab, Loose Wet, Loose Dry, Wind Slab, Wet Slab, Deep Slab, Cornice Fall, Glide); verify by ingesting a date with multiple problem types and confirming each is stored with likelihood/size/aspects/elevation_bands
- [ ] 3.3 Implement Open Avalanche Project fallback ingestion (2015-2021 coverage) with source='oap' tagging and schema mapping to CAIC format; verify by ingesting OAP data and confirming it populates caic_danger and caic_problems with source='oap'
- [ ] 3.4 Implement zone-to-station mapping via CAIC ArcGIS FeatureServer spatial join, handling stations in multiple zones; verify by querying zone_station_map and confirming at least one multi-zone station exists
- [ ] 3.5 Implement label temporal alignment (forecast valid date, not issuance date) and NULL handling for missing forecast days; verify by checking that a known gap date has NULL danger_level
- [ ] 3.6 Implement class distribution reporting that logs danger level and problem type counts after ingestion; verify by running ingestion and confirming the distribution summary is logged

## 4. Feature Engineering: Weather Rolling Windows

- [ ] 4.1 Implement 24h/48h/72h/96h/120h rolling window computation for temperature features (mean, min, max, std, range, trend) per station-date; verify by computing features for a known station-date and comparing temp_mean_24h against a hand-calculated value
- [ ] 4.2 Implement rolling window computation for precipitation features (sum, max_hourly, precip_hours, intensity) at all 5 window sizes; verify by computing for a known precipitation event and confirming precip_sum values match manual calculation
- [ ] 4.3 Implement rolling window computation for wind features (mean, max, std, direction_mode using circular statistics) at all 5 window sizes; verify by computing for a station-date with known wind data and confirming direction_mode handles circular wraparound correctly
- [ ] 4.4 Implement rolling window computation for SWE/snow depth features (swe_change, snow_depth_change, swe_rate, new_snow) at all 5 window sizes; verify by computing for a station-date spanning a storm event and confirming new_snow is clamped to zero when negative
- [ ] 4.5 Implement missing data handling: NULL all features when < 50% coverage in a window, include coverage_fraction when >= 50%; verify by computing features for a station-date with sparse data and confirming NULLs appear at the correct threshold
- [ ] 4.6 Implement incremental feature computation (recompute only dates within 120h of new observations, upsert to features_weather); verify by ingesting new data and confirming only affected dates are recomputed
- [ ] 4.7 Verify the complete weather feature vector produces exactly 90 features per station-date by running the full pipeline for one station-month and asserting column count

## 5. Feature Engineering: Physics Proxies

- [ ] 5.1 Implement temperature gradient days computation (daily gradient in K/m, cumulative faceting days since last reset event) with zero snow depth handling; verify by computing for a station with known thin snowpack and confirming gradient days accumulate and reset on rain-on-snow events
- [ ] 5.2 Implement surface hoar index (24h sum of clear_sky_indicator * RH_fraction for calm wind hours < 2 m/s); verify by computing for a known clear calm night and confirming index > 0, and for a windy night confirming index = 0
- [ ] 5.3 Implement wind slab loading index (24h sum of wind_speed * sin(wind_direction - aspect) * precip_rate) with DEM-derived aspect fallback; verify by computing for a known wind event with concurrent precipitation and confirming the value is physically reasonable
- [ ] 5.4 Implement rain-on-snow detection (precip > 0 AND temp > 32F AND snow_depth > 0) producing rain_on_snow_hours and rain_on_snow_amount, plus gradient reset when > 5mm in 24h; verify by injecting a known rain-on-snow event and confirming detection and gradient counter reset
- [ ] 5.5 Implement snow depth anomaly relative to 30-year climatological normal (from AWDB period-of-record stats or 10+ years of stored data); verify by computing for a station-date with known above-normal snow depth and confirming positive anomaly
- [ ] 5.6 Implement season context features: early_season_flag (before Jan 15 AND depth < 25th percentile) and freeze_thaw_cycles (14-day count of days crossing 32F both directions); verify with unit tests covering early-season thin snowpack and a known warm spell
- [ ] 5.7 Verify the complete physics proxy feature vector produces exactly 8 features per station-date by running the full pipeline and asserting column count

## 6. Data Alignment and Feature Matrix

- [ ] 6.1 Implement the feature matrix materialization join: weather features + physics proxies + station metadata, keyed on (station_id, date, elevation_band); verify by querying features_matrix and confirming 98+ columns with correct alignment
- [ ] 6.2 Implement label alignment: join features_matrix with caic_danger via zone_station_map on (zone_id → station_id, date, elevation_band); verify by querying the joined result and confirming labels are present for stations within zones that have forecasts
- [ ] 6.3 Create a data exploration notebook that produces summary statistics (feature distributions, label balance per elevation band, missing data heatmap, temporal coverage); verify by running the notebook end-to-end without errors

## 7. RF Baseline: Stage 1 Problem Type Classifier

- [ ] 7.1 Implement the temporal train/val/test split (train: 2015-01-01 to 2023-06-30, val: 2023-24 season, test: 2024-25 season) with summer gap exclusion and no temporal leakage in rolling window features; verify by asserting no date overlap between splits and that the earliest val-set rolling window does not reference training dates
- [ ] 7.2 Implement Stage 1 multi-label RF classifier (9 problem types, one model per elevation band) with SMOTE on training set and class_weight='balanced'; verify by training on a small data subset and confirming the model produces probability outputs for all 9 types
- [ ] 7.3 Implement out-of-fold Stage 1 predictions on training data to avoid label leakage into Stage 2; verify by confirming Stage 1 training-set predictions were generated via cross-validation, not direct prediction on training data

## 8. RF Baseline: Stage 2 Danger Level Classifier

- [ ] 8.1 Implement Stage 2 RF classifier using (weather + physics + Stage 1 probabilities) as input, producing danger level 1-5 with probability distribution, one model per elevation band; verify by training on a small data subset and confirming output includes 5-class probabilities
- [ ] 8.2 Implement per-class threshold tuning on validation set to optimize F1 per class; verify by confirming optimized thresholds differ from the default 0.5 for at least one class

## 9. RF Baseline: Evaluation and Tracking

- [ ] 9.1 Implement evaluation pipeline computing macro F1, per-class F1, ordinal accuracy (within +/-1), confusion matrix, and ROC-AUC per class, all reported per elevation band and aggregate; verify by running evaluation on the test set and confirming all metrics are computed without error
- [ ] 9.2 Implement benchmark comparison against a climatological baseline (most frequent class per season/elevation); verify by confirming RF metrics exceed the climatological baseline on macro F1
- [ ] 9.3 Implement SHAP integration: TreeSHAP values for all features per prediction, returning top 10 by absolute value with direction; verify by generating SHAP explanations for 10 test predictions and confirming feature names and contribution directions are present
- [ ] 9.4 Implement MLflow experiment logging (hyperparameters, feature list, split dates, all metrics, model artifact, SHAP summary plot); verify by confirming the MLflow UI shows the run with all logged artifacts
- [ ] 9.5 Create RF baseline notebook with end-to-end training, evaluation results, SHAP summary plots, and confusion matrices per elevation band; verify by running the notebook and confirming all cells execute without error

## 10. LSTM Sequence Model: Data Loader

- [ ] 10.1 Implement the hierarchical multi-rate PyTorch DataLoader: Branch 1 (7 days hourly, 168 steps), Branch 2 (30 days daily, 30 steps), Branch 3 (season 3-day windows, up to 60 steps) with per-station training-set normalization; verify by loading one batch and confirming three tensors with expected shapes and no NaN values
- [ ] 10.2 Implement missing data masking (binary mask concatenated as additional input feature, zero-fill for missing values); verify by loading a batch with known gaps and confirming mask correctly identifies missing positions
- [ ] 10.3 Implement weighted random sampler for balanced class representation in batches; verify by sampling 100 batches and confirming each danger level appears in approximately equal proportion

## 11. LSTM Sequence Model: Architecture and Training

- [ ] 11.1 Implement the 3-branch LSTM architecture (hourly branch with MaxPool + LSTM, daily branch LSTM, seasonal branch LSTM, concatenation, dense layers); verify by running a forward pass with random input and confirming output shape matches (batch_size, 5) for danger and (batch_size, 9) for problem types
- [ ] 11.2 Implement two-stage prediction heads: Stage 1 multi-label sigmoid for 9 problem types, Stage 2 softmax for danger level 1-5 taking concatenated features + Stage 1 probs; verify by confirming Stage 2 input dimension includes Stage 1 output
- [ ] 11.3 Implement joint training with weighted loss (focal loss gamma=2.0 for danger level head + multi-label BCE for problem type head); verify by training for 5 epochs on a small subset and confirming both loss components decrease
- [ ] 11.4 Implement elevation band handling (separate models or shared model with embedding, selected by validation performance comparison); verify by training both variants on a small subset and confirming the comparison metric is logged

## 12. LSTM Sequence Model: Evaluation and Comparison

- [ ] 12.1 Evaluate LSTM on the identical temporal test set as RF, reporting all the same metrics (macro F1, per-class F1, ordinal accuracy, confusion matrix, ROC-AUC, per elevation band); verify by confirming LSTM evaluation uses the same test samples as RF
- [ ] 12.2 Implement paired bootstrap significance test (p < 0.05) comparing LSTM vs RF on the test set; verify by running the test and confirming p-value is computed and reported
- [ ] 12.3 Implement ensemble prediction (weighted average of RF and LSTM probability distributions using inverse validation loss weights) with single-model fallback; verify by generating an ensemble prediction and confirming the output probability distribution sums to 1.0
- [ ] 12.4 Log LSTM runs to MLflow (architecture config, training curves, all metrics, model checkpoint, comparison table vs RF); verify by confirming the MLflow UI shows LSTM runs with comparison artifacts
- [ ] 12.5 Create LSTM experiment notebook with training curves, head-to-head comparison tables, and per-elevation-band analysis; verify by running the notebook end-to-end without errors

## 13. API: FastAPI Application

- [ ] 13.1 Implement FastAPI app skeleton with Pydantic request/response models for all endpoints (/predict, /history, /stations, /health) and consistent error format; verify by starting the app and confirming OpenAPI docs render at /docs with all endpoint schemas
- [ ] 13.2 Implement POST /predict endpoint: load model, query features from DuckDB, generate prediction with SHAP explanations, return danger level + probabilities + problem types + confidence; verify by sending a request for a known zone/band/date and confirming a valid prediction response
- [ ] 13.3 Implement GET /history endpoint with date range validation (max 365 days), returning predictions and actuals with prediction error; verify by querying a date range with known predictions and confirming the response includes prediction vs actual comparison
- [ ] 13.4 Implement GET /stations endpoint with optional zone_id filtering, returning station metadata and last observation date; verify by querying with and without zone_id filter and confirming correct station counts
- [ ] 13.5 Implement GET /health endpoint reporting status (healthy/degraded/unhealthy), last_ingestion_time, data_freshness_hours, model_version; verify by confirming healthy status on a fresh system and degraded status when ingestion is stale (> 36h)

## 14. API: Authentication, Scheduling, and Deployment

- [ ] 14.1 Implement API key authentication middleware (X-API-Key header, 401 on missing, /health exempt) and rate limiting (100 req/min per client, 429 with Retry-After); verify by sending requests without API key and confirming 401, and by exceeding rate limit and confirming 429
- [ ] 14.2 Implement daily ingestion scheduler (configurable time, default 06:00 UTC) running SNOTEL → CAIC → HRRR → features → predictions in order, with failure isolation per stage; verify by triggering a manual run via POST /ingest and confirming all stages execute and failures in one stage do not block subsequent independent stages
- [ ] 14.3 Implement prediction caching in DuckDB predictions table (cache key: zone+band+date+model_version) and historical prediction storage for monitoring; verify by generating a prediction twice and confirming the second call returns the cached result
- [ ] 14.4 Create Dockerfile (single-stage build, health check on /health, env vars for DUCKDB_PATH/MODEL_PATH/API_KEY, startup within 30s) and verify by building the image and running the container with a test API key
- [ ] 14.5 Create API integration tests covering the golden path (ingest → features → predict → history) and error cases (missing API key, invalid zone, stale data); verify by running the test suite and confirming all tests pass

## 15. Spatial Downscaling: HRRR Ingestion

- [ ] 15.1 Implement HrrrClient using Herbie to extract HRRR analysis grids for the Colorado domain, storing interpolated values at SNOTEL station locations in hrrr_station and full grids in hrrr_grid (partitioned by date); verify by ingesting 3 days and confirming rows in both tables
- [ ] 15.2 Implement SNOTEL-based precipitation bias correction (rolling 30-day ratio of SNOTEL observed to HRRR predicted, with IDW for non-station gridpoints); verify by comparing corrected vs raw HRRR precipitation at a known station and confirming the bias is reduced
- [ ] 15.3 Implement watermark-based incremental HRRR ingestion consistent with SNOTEL client pattern; verify by running incremental ingestion twice and confirming the second run requests only new data

## 16. Spatial Downscaling: Terrain Features and XGBoost Model

- [ ] 16.1 Implement DEM terrain feature extraction from USGS 3DEP (elevation, slope, aspect, TPI@500m, SVF, curvature) for arbitrary points within the Colorado domain (37-41N, 105-108W), with domain boundary validation; verify by computing terrain features for 3 known SNOTEL station locations and confirming elevation matches station metadata within 50m
- [ ] 16.2 Implement XGBoost downscaling model training: inputs are (HRRR interpolated values + elevation difference + terrain features + HRRR spatial gradients), targets are SNOTEL observations; verify by training on a subset and confirming RMSE improves over raw HRRR values
- [ ] 16.3 Implement leave-one-station-out cross-validation for spatial generalization evaluation; verify by running LOSO-CV and confirming per-station metrics are reported
- [ ] 16.4 Implement cold air pool correction (TPI@500m < -50m, wind < 3 m/s, clear sky → temperature reduction scaling with TPI magnitude); verify by predicting temperature at a known valley station and confirming the correction reduces the HRRR warm bias
- [ ] 16.5 Implement downscaled output in station-compatible schema so weather window and physics proxy pipelines process it identically to station data; verify by running the feature engineering pipeline on downscaled output and confirming it produces the same 98+ feature columns
- [ ] 16.6 Implement uncertainty quantification (prediction intervals via quantile regression or conformal prediction) for all downscaled variables; verify by generating predictions and confirming 10th/90th percentile bounds bracket the observed values at held-out stations at approximately the expected rate

## 17. Spatial Downscaling: Evaluation

- [ ] 17.1 Run TopoPyScale benchmark for the same stations and time period; verify by confirming TopoPyScale output covers the evaluation set
- [ ] 17.2 Produce comparison report (RMSE, MAE, bias, correlation for temperature, precipitation, wind speed) between XGBoost downscaling and TopoPyScale at held-out stations; verify by confirming all metrics are computed and the comparison table is logged to MLflow
- [ ] 17.3 Create downscaling evaluation notebook with maps of station-level errors, scatter plots of predicted vs observed, and spatial error patterns; verify by running the notebook end-to-end without errors

## 18. Full Integration

- [ ] 18.1 Connect downscaling output to the instability prediction pipeline: downscaled weather at arbitrary locations → weather windows → physics proxies → feature matrix → model prediction; verify by generating a prediction at a non-station location and confirming the output includes danger level, problem types, and SHAP explanations
- [ ] 18.2 Integrate CAIC avalanche path geometries from ArcGIS FeatureServer as spatial prediction targets; verify by querying path geometries and confirming they can be used as prediction request locations
- [ ] 18.3 Extend the FastAPI /predict endpoint to accept latitude/longitude in addition to zone_id, returning terrain-resolved predictions using the downscaling pipeline; verify by sending a lat/lon prediction request and confirming a valid response with the same schema as zone-based predictions
- [ ] 18.4 Implement end-to-end evaluation: predictions at SNOTEL station locations via the spatial path vs direct station-based predictions, confirming consistency; verify by comparing predictions from both paths for 10 stations and confirming danger levels agree within 1 level for > 90% of cases
