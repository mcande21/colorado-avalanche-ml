# colorado-avalanche-ml

Snowpack instability prediction from weather sequences for Colorado avalanche terrain.

## Build & Run

```bash
# Install dependencies
pip install -e ".[dev,ingestion]"

# Run tests
pytest tests/

# Lint
ruff check src/

# Run API server
uvicorn src.api.app:app --reload

# Run ingestion
python -m src.ingestion.scheduler
```

## Architecture

Two-stage ML pipeline: predict avalanche problem types (Persistent Slab, Storm Slab, Loose Wet), then danger level per elevation band.

- **Model A:** Random Forest baseline with 67-88 engineered features + physics proxies
- **Model B:** LSTM with hierarchical multi-rate temporal input (7-day hourly / 30-day daily / season 3-day)
- **API:** FastAPI serving predictions per CAIC zone and elevation band
- **Data:** SNOTEL (AWDB REST API), CAIC labels (caic-python + OAP), HRRR grids (AWS), USGS DEMs

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/ingestion/` | Data ingestion clients (SNOTEL, CAIC, HRRR) |
| `src/features/` | Feature engineering (weather windows, physics proxies, season context) |
| `src/models/` | Model implementations (RF two-stage, LSTM hierarchical) |
| `src/api/` | FastAPI service |
| `src/db/` | DuckDB feature store |
| `data/` | Raw + processed data (gitignored except terrain) |
| `notebooks/` | Exploration and experiment notebooks |

## Conventions

- Python 3.11+, ruff for linting (line-length: 100)
- No trailing whitespace
- Parquet for data storage, DuckDB for queries
- MLflow for experiment tracking
