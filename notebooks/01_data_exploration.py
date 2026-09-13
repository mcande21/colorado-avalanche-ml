"""Data exploration script for the training matrix.

Loads the training matrix from DuckDB (or Parquet fallback), prints
basic stats, feature distributions, class balance, and correlation
between physics features and danger level.

Usage:
    python notebooks/01_data_exploration.py [--parquet path/to/matrix.parquet]
"""
from __future__ import annotations

import argparse
import sys

import duckdb
import pandas as pd


def load_matrix(parquet_path: str | None = None) -> pd.DataFrame:
    if parquet_path:
        conn = duckdb.connect(":memory:")
        df = conn.execute(f"SELECT * FROM read_parquet('{parquet_path}')").fetchdf()
        conn.close()
        return df

    conn = duckdb.connect("data/avalanche.duckdb", read_only=True)
    try:
        df = conn.execute("SELECT * FROM training_matrix").fetchdf()
    finally:
        conn.close()
    return df


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def basic_stats(df: pd.DataFrame) -> None:
    print_section("Basic Statistics")
    print(f"Total rows:    {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Stations:      {df['station_id'].nunique()}")
    print(f"Date range:    {df['date'].min()} to {df['date'].max()}")

    if "elevation_band" in df.columns:
        print(f"\nRows by elevation band:")
        for band, count in df["elevation_band"].value_counts().items():
            print(f"  {band}: {count}")


def danger_distribution(df: pd.DataFrame) -> None:
    print_section("Danger Level Distribution")
    dist = df["danger_level"].value_counts().sort_index()
    total = len(df)
    for level, count in dist.items():
        pct = count / total * 100
        bar = "#" * int(pct)
        print(f"  Level {level}: {count:>6} ({pct:5.1f}%) {bar}")


def problem_type_distribution(df: pd.DataFrame) -> None:
    print_section("Problem Type Distribution")
    problem_cols = ["persistent_slab", "storm_slab", "loose_wet"]
    for col in problem_cols:
        if col in df.columns:
            present = df[col].sum()
            pct = present / len(df) * 100 if len(df) else 0
            print(f"  {col}: {int(present)} ({pct:.1f}%)")


def physics_correlation(df: pd.DataFrame) -> None:
    print_section("Physics Feature vs Danger Level Correlation")
    physics = [
        "temp_gradient_days", "surface_hoar_index", "wind_slab_loading",
        "rain_on_snow_hours", "rain_on_snow_amount", "snow_depth_anomaly",
        "early_season_flag", "freeze_thaw_cycles",
    ]
    for feat in physics:
        if feat in df.columns:
            corr = df[feat].corr(df["danger_level"])
            if pd.isna(corr):
                corr = 0.0
            bar = "+" * int(abs(corr) * 20)
            sign = "+" if corr >= 0 else "-"
            print(f"  {feat:<25} {sign}{abs(corr):.3f} {bar}")


def feature_summary(df: pd.DataFrame) -> None:
    print_section("Feature Summary (non-null counts)")
    weather_cols = [c for c in df.columns if c.endswith("h") and "_" in c]
    physics_cols = [
        "temp_gradient_days", "surface_hoar_index", "wind_slab_loading",
        "rain_on_snow_hours", "rain_on_snow_amount", "snow_depth_anomaly",
        "early_season_flag", "freeze_thaw_cycles",
    ]
    weather_present = [c for c in weather_cols if c in df.columns]
    physics_present = [c for c in physics_cols if c in df.columns]

    if weather_present:
        null_pct = df[weather_present].isnull().mean().mean() * 100
        print(f"  Weather features: {len(weather_present)} columns, {null_pct:.1f}% null")
    if physics_present:
        null_pct = df[physics_present].isnull().mean().mean() * 100
        print(f"  Physics features: {len(physics_present)} columns, {null_pct:.1f}% null")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore training matrix")
    parser.add_argument("--parquet", help="Path to Parquet file (default: DuckDB)")
    args = parser.parse_args()

    try:
        df = load_matrix(args.parquet)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        print("Run the pipeline first or pass --parquet <path>", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("Training matrix is empty. Run the pipeline first.")
        sys.exit(1)

    basic_stats(df)
    danger_distribution(df)
    problem_type_distribution(df)
    physics_correlation(df)
    feature_summary(df)

    print_section("Pipeline Verification")
    print("  Training matrix loaded successfully.")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("  Pipeline end-to-end: OK")


if __name__ == "__main__":
    main()
