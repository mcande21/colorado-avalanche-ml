# %% [markdown]
# # Model Comparison: Random Forest vs Hierarchical LSTM
#
# Head-to-head comparison of both models on snowpack instability prediction.
# Requires completed MLflow runs from both RF and LSTM training pipelines.

# %%
import mlflow
import numpy as np
import pandas as pd

from avalanche_ml.models.comparison import (
    compare_models,
    comparison_report,
    model_recommendation,
)
from avalanche_ml.models.tracking import configure_tracking

# %% [markdown]
# ## Load MLflow Experiments

# %%
TRACKING_DIR = "../data/mlruns"
configure_tracking(TRACKING_DIR)

rf_experiment = mlflow.get_experiment_by_name("snowpack-instability-rf")
lstm_experiment = mlflow.get_experiment_by_name("snowpack-instability-lstm")

print(f"RF experiment: {rf_experiment.experiment_id if rf_experiment else 'NOT FOUND'}")
print(f"LSTM experiment: {lstm_experiment.experiment_id if lstm_experiment else 'NOT FOUND'}")

# %%
client = mlflow.tracking.MlflowClient()

rf_runs = client.search_runs(
    experiment_ids=[rf_experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1,
) if rf_experiment else []

lstm_runs = client.search_runs(
    experiment_ids=[lstm_experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1,
) if lstm_experiment else []

if not rf_runs or not lstm_runs:
    raise RuntimeError(
        "Need at least one completed run from each experiment. "
        f"RF runs: {len(rf_runs)}, LSTM runs: {len(lstm_runs)}"
    )

rf_run_id = rf_runs[0].info.run_id
lstm_run_id = lstm_runs[0].info.run_id
print(f"RF run: {rf_run_id}")
print(f"LSTM run: {lstm_run_id}")

# %% [markdown]
# ## Comparison Table

# %%
results = compare_models(rf_run_id, lstm_run_id)

rows = []
for metric in sorted(results["rf_metrics"]):
    rf_val = results["rf_metrics"].get(metric, float("nan"))
    lstm_val = results["lstm_metrics"].get(metric, float("nan"))
    diff = results["differences"].get(metric, float("nan"))
    winner = results["winner_per_metric"].get(metric, "")
    rows.append({
        "Metric": metric,
        "RF": round(rf_val, 4),
        "LSTM": round(lstm_val, 4),
        "Diff (LSTM-RF)": round(diff, 4),
        "Winner": winner,
    })

comparison_df = pd.DataFrame(rows)
print(comparison_df.to_string(index=False))

# %% [markdown]
# ## Binary F1 per Elevation Band

# %%
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    band_data = results.get("per_elevation_band", {})
    if band_data:
        bands = sorted(band_data.keys())
        rf_f1s = [band_data[b].get("rf", {}).get("macro_f1", 0) for b in bands]
        lstm_f1s = [band_data[b].get("lstm", {}).get("macro_f1", 0) for b in bands]

        x = np.arange(len(bands))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width / 2, rf_f1s, width, label="RF", color="#2196F3")
        ax.bar(x + width / 2, lstm_f1s, width, label="LSTM", color="#FF9800")
        ax.set_xlabel("Elevation Band")
        ax.set_ylabel("Macro F1")
        ax.set_title("Model Performance by Elevation Band")
        ax.set_xticks(x)
        ax.set_xticklabels(bands, rotation=15)
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig("../data/comparison_elevation_bands.png", dpi=150)
        plt.show()
    else:
        print("No per-elevation-band data available.")
except ImportError:
    print("matplotlib not available for plotting.")

# %% [markdown]
# ## Confusion Matrices Side by Side

# %%
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rf_run = client.get_run(rf_run_id)
    lstm_run = client.get_run(lstm_run_id)

    print("Confusion matrix visualization requires saved artifacts.")
    print("RF metrics:", {k: v for k, v in rf_run.data.metrics.items()
                          if "confusion" not in k})
    print("LSTM metrics:", {k: v for k, v in lstm_run.data.metrics.items()
                            if "confusion" not in k})
except ImportError:
    print("matplotlib not available.")

# %% [markdown]
# ## LSTM Training Loss Curves

# %%
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Training loss curves require the history dict from run_lstm_training_pipeline.")
    print("Load from a saved artifact or re-run the pipeline to access.")
except ImportError:
    print("matplotlib not available.")

# %% [markdown]
# ## Recommendation

# %%
rec = model_recommendation(results)
print(f"\nRecommendation: {rec['recommendation'].upper()}")
print(f"Rationale: {rec['rationale']}")

# %%
print("\n" + comparison_report(results))
