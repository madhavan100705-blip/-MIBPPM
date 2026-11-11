
"""
MIBPPM - SBPI Model Visualization
Generates:
1. Model MAE comparison graphs
2. Actual vs Predicted graphs
3. Residual analysis graphs
4. Residual distribution graphs

Input:
results/sbpi/model_metrics.csv
results/sbpi/oof_predictions.csv

Output:
results/sbpi/graphs/
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results" / "sbpi"
PLOT_DIR = RESULTS_DIR / "graphs"

METRICS_FILE = RESULTS_DIR / "model_metrics.csv"
PREDICTIONS_FILE = RESULTS_DIR / "oof_predictions.csv"


# --------------------------------------------------
# 2. CREATE OUTPUT DIRECTORY
# --------------------------------------------------

PLOT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. SAFE FILENAME FUNCTION
# --------------------------------------------------

def safe_filename(name):
    """
    Convert target names into Windows-safe filenames.

    Example:
    PPV (mm/s) -> PPV_mm_s
    """

    name = str(name)

    name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        name
    )

    return name.strip("_")


# --------------------------------------------------
# 4. CHECK INPUT FILES
# --------------------------------------------------

if not METRICS_FILE.exists():
    print("ERROR: Model metrics file not found:")
    print(METRICS_FILE)
    sys.exit(1)

if not PREDICTIONS_FILE.exists():
    print("ERROR: OOF predictions file not found:")
    print(PREDICTIONS_FILE)
    sys.exit(1)


# --------------------------------------------------
# 5. LOAD RESULTS
# --------------------------------------------------

print("\nLoading SBPI experiment results...")

metrics = pd.read_csv(METRICS_FILE)

predictions = pd.read_csv(PREDICTIONS_FILE)

required_metrics = {
    "Target", "Model", "MAE", "RMSE", "R2"
}

required_predictions = {
    "Target", "Model", "Actual", "Predicted"
}

if not required_metrics.issubset(metrics.columns):
    raise ValueError(
        "Missing columns in model_metrics.csv: "
        f"{required_metrics - set(metrics.columns)}"
    )

if not required_predictions.issubset(predictions.columns):
    raise ValueError(
        "Missing columns in oof_predictions.csv: "
        f"{required_predictions - set(predictions.columns)}"
    )

targets = metrics["Target"].dropna().unique()

print("Targets found:", list(targets))


# --------------------------------------------------
# 6. GRAPH 1 - MODEL MAE COMPARISON
# --------------------------------------------------

print("\nGenerating model comparison graphs...")

for target in targets:

    subset = metrics[
        metrics["Target"] == target
    ].sort_values("MAE")

    filename = safe_filename(target)

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        subset["Model"],
        subset["MAE"]
    )

    plt.title(
        f"Model MAE Comparison - {target}",
        fontsize=14
    )

    plt.xlabel("Machine Learning Model")
    plt.ylabel("Mean Absolute Error (MAE)")

    plt.xticks(
        rotation=35,
        ha="right"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    # Display MAE above each bar.
    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.tight_layout()

    output_file = (
        PLOT_DIR /
        f"{filename}_model_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_file.name)


# --------------------------------------------------
# 7. SELECT BEST MODEL FOR EACH TARGET
# --------------------------------------------------

best_models = (
    metrics
    .sort_values("MAE")
    .groupby("Target", sort=False)
    .first()
    .reset_index()
)

best_models.to_csv(
    PLOT_DIR / "best_model_summary.csv",
    index=False
)


# --------------------------------------------------
# 8. GRAPH 2 - ACTUAL VS PREDICTED
# --------------------------------------------------

print("\nGenerating actual vs predicted graphs...")

for _, row in best_models.iterrows():

    target = row["Target"]
    model_name = row["Model"]

    subset = predictions[
        (predictions["Target"] == target) &
        (predictions["Model"] == model_name)
    ].copy()

    if subset.empty:
        print("Skipping empty prediction set:", target)
        continue

    actual = subset["Actual"].to_numpy(dtype=float)
    predicted = subset["Predicted"].to_numpy(dtype=float)

    filename = safe_filename(target)

    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())

    if np.isclose(lower, upper):
        lower -= 1
        upper += 1

    plt.figure(figsize=(7, 7))

    plt.scatter(
        actual,
        predicted,
        alpha=0.8
    )

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Ideal prediction"
    )

    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")

    plt.title(
        f"Actual vs Predicted - {target}\n"
        f"Model: {model_name}"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    output_file = (
        PLOT_DIR /
        f"{filename}_actual_vs_predicted.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_file.name)


# --------------------------------------------------
# 9. GRAPH 3 - RESIDUAL ANALYSIS
# --------------------------------------------------

print("\nGenerating residual analysis graphs...")

for _, row in best_models.iterrows():

    target = row["Target"]
    model_name = row["Model"]

    subset = predictions[
        (predictions["Target"] == target) &
        (predictions["Model"] == model_name)
    ].copy()

    if subset.empty:
        continue

    actual = subset["Actual"].to_numpy(dtype=float)
    predicted = subset["Predicted"].to_numpy(dtype=float)

    residuals = predicted - actual

    filename = safe_filename(target)

    plt.figure(figsize=(8, 5))

    plt.scatter(
        predicted,
        residuals,
        alpha=0.8
    )

    plt.axhline(
        y=0,
        linestyle="--",
        label="Zero residual"
    )

    plt.xlabel("Predicted Values")
    plt.ylabel("Residual (Predicted - Actual)")

    plt.title(
        f"Residual Analysis - {target}"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    output_file = (
        PLOT_DIR /
        f"{filename}_residuals.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_file.name)


# --------------------------------------------------
# 10. GRAPH 4 - RESIDUAL DISTRIBUTION
# --------------------------------------------------

print("\nGenerating residual distribution graphs...")

for _, row in best_models.iterrows():

    target = row["Target"]
    model_name = row["Model"]

    subset = predictions[
        (predictions["Target"] == target) &
        (predictions["Model"] == model_name)
    ].copy()

    if subset.empty:
        continue

    residuals = (
        subset["Predicted"].to_numpy(dtype=float) -
        subset["Actual"].to_numpy(dtype=float)
    )

    filename = safe_filename(target)

    plt.figure(figsize=(8, 5))

    plt.hist(
        residuals,
        bins=10,
        edgecolor="black"
    )

    plt.axvline(
        0,
        linestyle="--",
        label="Zero error"
    )

    plt.xlabel("Residual")
    plt.ylabel("Frequency")

    plt.title(
        f"Residual Distribution - {target}"
    )

    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_file = (
        PLOT_DIR /
        f"{filename}_residual_distribution.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_file.name)


# --------------------------------------------------
# 11. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("SBPI VISUALIZATION COMPLETED")
print("=" * 60)

print("\nBEST MODEL FOR EACH TARGET")

print(
    best_models[
        ["Target", "Model", "MAE", "RMSE", "R2"]
    ].to_string(index=False)
)

print("\nTotal targets:", len(targets))

print("Graphs saved in:")
print(PLOT_DIR.resolve())

print("\nAll graph generation tasks completed successfully.")
