## Why

Colorado avalanche forecasting relies on expert judgment applied to sparse weather station data, with no operational ML system predicting instability from weather sequences and terrain features. Published research (Schwartzreich & Rodriguez 2026, Swiss SLF) demonstrates that two-stage classification — predicting avalanche problem types first, then danger level — achieves operational accuracy, but no open implementation exists for Colorado-specific terrain and data sources. Building this now leverages newly available SNOTEL hourly data, the Open Avalanche Project labeled dataset (2015-2021), and HRRR 3km grids on AWS to create a pipeline that predicts avalanche danger by problem type and elevation band, served via API with spatial downscaling for micro-weather estimation in complex terrain.

## What Changes

- Ingest SNOTEL station data (118 CO stations) via AWDB REST API into a DuckDB feature store
- Ingest CAIC danger ratings via caic-python with Open Avalanche Project fallback labels
- Engineer 67-88 features: weather rolling windows (24h-120h), physics proxies (temp gradient days, surface hoar index, wind slab loading, rain-on-snow, snow depth anomaly)
- Train two-stage Random Forest baseline (predict problem types -> predict danger level) with SHAP explanations
- Train hierarchical multi-rate LSTM (180-day lookback, N-HiTS-style multi-rate MaxPool) and benchmark against RF
- Predict per elevation band (above/near/below treeline) with binary + probability distribution output
- Serve predictions via FastAPI with daily data ingestion pipeline
- Track experiments with MLflow, store features in DuckDB
- Spatial downscaling: HRRR 3km grids + SNOTEL bias correction + DEM terrain features for micro-weather at arbitrary locations
- Handle cold air pool dynamics in spatial downscaling

## Capabilities

### New Capabilities

- `data/snotel-ingestion`: Ingest and store SNOTEL station observations (118 CO stations, hourly/daily) from AWDB REST API into DuckDB
- `data/caic-labels`: Ingest CAIC avalanche danger ratings and problem type labels, with Open Avalanche Project as fallback source
- `data/hrrr-ingestion`: Ingest HRRR 3km gridded weather forecasts from AWS via Herbie, with precipitation bias correction
- `features/weather-windows`: Rolling window feature engineering over weather time series (24h, 48h, 72h, 120h aggregations)
- `features/physics-proxies`: Physics-informed proxy features approximating SNOWPACK output (temp gradient days, surface hoar index, wind slab loading, rain-on-snow, snow depth anomaly)
- `models/two-stage-rf`: Two-stage Random Forest classifier (problem type prediction -> danger level prediction) with SHAP explanations
- `models/hierarchical-lstm`: Hierarchical multi-rate LSTM with 180-day lookback and N-HiTS-style temporal pooling
- `prediction/elevation-band`: Per-elevation-band prediction (above/near/below treeline) with binary and probability distribution output
- `spatial/downscaling`: Spatial downscaling combining HRRR grids, SNOTEL bias correction, DEM terrain features, and cold air pool handling
- `api/prediction-service`: FastAPI prediction service with daily ingestion pipeline and Docker deployment

### Modified Capabilities

_None — greenfield project, no existing specs._

## Impact

- **New code**: Full Python package — data ingestion, feature engineering, model training, API serving, spatial downscaling
- **External APIs**: AWDB REST (no auth), CAIC scraping (fragile, #1 risk), HRRR on AWS (no auth), USGS 3DEP DEM, CAIC ArcGIS FeatureServer, CDOT RWIS
- **Dependencies**: Python 3.11+, scikit-learn, PyTorch, FastAPI, DuckDB, MLflow, Herbie, snotelpy, caic-python, SHAP, Polars/Pandas
- **Data storage**: DuckDB feature store (local), MLflow tracking store
- **Deployment**: Docker container, daily cron ingestion
- **Key risks**: CAIC data access (scraping, no official API — fallback to OAP labels); HRRR 25-65% precipitation bias (mandatory correction); severe class imbalance (1.1% avalanche days — SMOTE/cost-sensitive required); DL hasn't outperformed RF in published avalanche studies (RF baseline first)
