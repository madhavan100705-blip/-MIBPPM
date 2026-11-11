
"""
MIBPPM - EXPERIMENT 8
Detailed validation of SBPI AI models.

Analyses:
1. Residual statistics for all five targets.
2. Error analysis by Blast ID.
3. Residual vs predicted plots.
4. Q-Q plots.
5. Residual distribution plots.
6. Shapiro-Wilk normality test.
7. Large-error identification.

Uses out-of-fold predictions from Experiment 7.
Does not retrain models or use training-set predictions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


# ==================================================
# 1. PATH CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results" / "sbpi"

OUTPUT_DIR = RESULTS_DIR / "experiment8"

GRAPH_DIR = OUTPUT_DIR / "graphs"

PREDICTION_FILE = RESULTS_DIR / "oof_predictions.csv"
METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. SAFE FILENAMES
# ==================================================

def safe_filename(name):

    import re

    return re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(name)
    ).strip("_")


# ==================================================
# 3. LOAD DATA
# ==================================================

print("\n" + "=" * 65)
print("MIBPPM - EXPERIMENT 8: MODEL VALIDATION")
print("=" * 65)

if not PREDICTION_FILE.exists():
    raise FileNotFoundError(PREDICTION_FILE)

if not METRICS_FILE.exists():
    raise FileNotFoundError(METRICS_FILE)

predictions = pd.read_csv(PREDICTION_FILE)
metrics = pd.read_csv(METRICS_FILE)

required_columns = [
    "Blast ID",
    "Target",
    "Model",
    "Actual",
    "Predicted"
]

missing = [
    col for col in required_columns
    if col not in predictions.columns
]

if missing:
    raise ValueError(
        f"Missing prediction columns: {missing}"
    )

predictions = predictions.dropna(
    subset=required_columns
).copy()

predictions["Residual"] = (
    predictions["Predicted"] -
    predictions["Actual"]
)

predictions["Absolute_Error"] = (
    predictions["Residual"].abs()
)

print("\nDATA SUMMARY")
print("-" * 50)

print("Prediction records:", len(predictions))
print("Distinct Blast IDs:", predictions["Blast ID"].nunique())

print("\nTargets:")
for target in predictions["Target"].unique():
    print("-", target)


# ==================================================
# 4. SELECT BEST MODEL FOR EACH TARGET
# ==================================================

best_models = (
    metrics
    .sort_values("MAE")
    .groupby("Target")
    .first()
    .reset_index()
)

best_models.to_csv(
    OUTPUT_DIR / "best_model_selection.csv",
    index=False
)


# ==================================================
# 5. CONTAINERS FOR RESULTS
# ==================================================

overall_results = []
blast_results = []
outlier_results = []
normality_results = []

diagnostic_predictions = []


# ==================================================
# 6. ANALYSE EACH TARGET
# ==================================================

for _, best in best_models.iterrows():

    target = best["Target"]
    model_name = best["Model"]

    print("\n" + "=" * 65)
    print("TARGET:", target)
    print("MODEL:", model_name)
    print("=" * 65)

    subset = predictions[
        (predictions["Target"] == target) &
        (predictions["Model"] == model_name)
    ].copy()

    if subset.empty:
        print("No predictions found. Skipping.")
        continue

    actual = subset["Actual"].to_numpy(dtype=float)
    predicted = subset["Predicted"].to_numpy(dtype=float)
    residual = predicted - actual
    abs_error = np.abs(residual)

    n = len(subset)

    # ----------------------------------------------
    # A. Overall residual statistics
    # ----------------------------------------------

    mean_error = np.mean(residual)
    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean(residual ** 2))
    residual_std = np.std(residual, ddof=1)

    median_error = np.median(residual)
    max_abs_error = np.max(abs_error)
    min_error = np.min(residual)
    max_error = np.max(residual)

    # Relative MAE, only meaningful for nonzero
    # mean absolute actual values.
    mean_actual_abs = np.mean(np.abs(actual))

    relative_mae = (
        mae / mean_actual_abs * 100
        if mean_actual_abs > 0
        else np.nan
    )

    overall_results.append({
        "Target": target,
        "Best_Model": model_name,
        "N": n,
        "Mean_Residual": mean_error,
        "Median_Residual": median_error,
        "MAE": mae,
        "RMSE": rmse,
        "Residual_Std": residual_std,
        "Minimum_Residual": min_error,
        "Maximum_Residual": max_error,
        "Maximum_Absolute_Error": max_abs_error,
        "Relative_MAE_Percent": relative_mae
    })

    subset["Residual"] = residual
    subset["Absolute_Error"] = abs_error

    diagnostic_predictions.append(subset)

    print("\nOVERALL ERROR STATISTICS")
    print(f"Mean residual: {mean_error:.5f}")
    print(f"Median residual: {median_error:.5f}")
    print(f"MAE: {mae:.5f}")
    print(f"RMSE: {rmse:.5f}")
    print(f"Residual standard deviation: {residual_std:.5f}")
    print(f"Maximum absolute error: {max_abs_error:.5f}")
    print(f"Relative MAE: {relative_mae:.3f}%")

    # ----------------------------------------------
    # B. Residual vs predicted graph
    # ----------------------------------------------

    filename = safe_filename(target)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        predicted,
        residual,
        alpha=0.8
    )

    plt.axhline(
        0,
        linestyle="--",
        label="Zero residual"
    )

    plt.axhline(
        mean_error,
        linestyle=":",
        label="Mean residual"
    )

    plt.xlabel("Predicted Value")
    plt.ylabel("Residual (Predicted - Actual)")

    plt.title(
        f"Residual vs Predicted - {target}"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / f"{filename}_residual_vs_predicted.png",
        dpi=200
    )

    plt.close()

    # ----------------------------------------------
    # C. Q-Q plot
    # ----------------------------------------------

    plt.figure(figsize=(7, 6))

    stats.probplot(
        residual,
        dist="norm",
        plot=plt
    )

    plt.title(
        f"Q-Q Plot - {target}"
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / f"{filename}_qq_plot.png",
        dpi=200
    )

    plt.close()

    # ----------------------------------------------
    # D. Residual distribution
    # ----------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        residual,
        bins=min(10, max(5, n // 3)),
        edgecolor="black"
    )

    plt.axvline(
        0,
        linestyle="--",
        label="Zero residual"
    )

    plt.axvline(
        mean_error,
        linestyle=":",
        label="Mean residual"
    )

    plt.xlabel("Residual")
    plt.ylabel("Frequency")

    plt.title(
        f"Residual Distribution - {target}"
    )

    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / f"{filename}_residual_distribution.png",
        dpi=200
    )

    plt.close()

    # ----------------------------------------------
    # E. Shapiro-Wilk normality test
    # ----------------------------------------------

    if 3 <= n <= 5000:

        shapiro_stat, shapiro_p = stats.shapiro(
            residual
        )

    else:

        shapiro_stat = np.nan
        shapiro_p = np.nan

    normality_results.append({
        "Target": target,
        "Shapiro_Statistic": shapiro_stat,
        "Shapiro_P_Value": shapiro_p,
        "N": n,
        "Interpretation": (
            "Evidence against normality at 5% level"
            if not np.isnan(shapiro_p) and shapiro_p < 0.05
            else "No significant evidence against normality"
            if not np.isnan(shapiro_p)
            else "Test not performed"
        )
    })

    print("\nSHAPIRO-WILK TEST")
    print("Statistic:", shapiro_stat)
    print("P-value:", shapiro_p)

    # Note: This test is exploratory, not a
    # requirement for good predictive performance.

    # ----------------------------------------------
    # F. Error analysis by Blast ID
    # ----------------------------------------------

    grouped = subset.groupby("Blast ID")

    for blast_id, group in grouped:

        group_residual = group["Residual"].to_numpy()
        group_abs_error = group["Absolute_Error"].to_numpy()

        blast_results.append({
            "Target": target,
            "Model": model_name,
            "Blast_ID": blast_id,
            "Records": len(group),
            "Mean_Residual": np.mean(group_residual),
            "MAE": np.mean(group_abs_error),
            "RMSE": np.sqrt(
                np.mean(group_residual ** 2)
            ),
            "Maximum_Absolute_Error": np.max(
                group_abs_error
            )
        })

    # ----------------------------------------------
    # G. Identify unusually large errors
    # ----------------------------------------------

    q1 = np.quantile(abs_error, 0.25)
    q3 = np.quantile(abs_error, 0.75)

    iqr = q3 - q1

    upper_limit = q3 + 1.5 * iqr

    outliers = subset[
        subset["Absolute_Error"] > upper_limit
    ]

    for _, item in outliers.iterrows():

        outlier_results.append({
            "Target": target,
            "Model": model_name,
            "Blast_ID": item["Blast ID"],
            "Actual": item["Actual"],
            "Predicted": item["Predicted"],
            "Residual": item["Residual"],
            "Absolute_Error": item["Absolute_Error"],
            "IQR_Upper_Limit": upper_limit
        })

    print("\nLARGE ERROR OBSERVATIONS")
    print(
        f"Outlier threshold (IQR rule): {upper_limit:.5f}"
    )
    print("Observations above threshold:", len(outliers))


# ==================================================
# 7. SAVE CSV REPORTS
# ==================================================

overall_df = pd.DataFrame(overall_results)

blast_df = pd.DataFrame(blast_results)

normality_df = pd.DataFrame(normality_results)

outlier_df = pd.DataFrame(outlier_results)

prediction_df = pd.concat(
    diagnostic_predictions,
    ignore_index=True
)

overall_df.to_csv(
    OUTPUT_DIR / "overall_diagnostics.csv",
    index=False
)

blast_df.to_csv(
    OUTPUT_DIR / "blast_id_error_analysis.csv",
    index=False
)

normality_df.to_csv(
    OUTPUT_DIR / "normality_tests.csv",
    index=False
)

outlier_df.to_csv(
    OUTPUT_DIR / "large_error_observations.csv",
    index=False
)

prediction_df.to_csv(
    OUTPUT_DIR / "best_model_oof_diagnostics.csv",
    index=False
)


# ==================================================
# 8. BLAST ID ERROR COMPARISON GRAPH
# ==================================================

for target in blast_df["Target"].unique():

    subset = blast_df[
        blast_df["Target"] == target
    ].sort_values("Blast_ID")

    filename = safe_filename(target)

    plt.figure(figsize=(11, 5))

    plt.bar(
        subset["Blast_ID"].astype(str),
        subset["MAE"]
    )

    plt.xlabel("Blast ID")
    plt.ylabel("Mean Absolute Error")

    plt.title(
        f"Prediction Error by Blast ID - {target}"
    )

    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / f"{filename}_blast_id_error.png",
        dpi=200
    )

    plt.close()


# ==================================================
# 9. FINAL REPORT
# ==================================================

print("\n" + "=" * 65)
print("EXPERIMENT 8 COMPLETED")
print("=" * 65)

print("\nOVERALL MODEL DIAGNOSTICS")

if not overall_df.empty:

    print(
        overall_df[
            ["Target", "MAE", "RMSE", "Mean_Residual",
             "Maximum_Absolute_Error"]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

print("\nBLAST ID ERROR SUMMARY")

if not blast_df.empty:

    print(
        blast_df.sort_values(
            ["Target", "MAE"],
            ascending=[True, False]
        ).groupby("Target").head(3)[
            ["Target", "Blast_ID", "MAE", "RMSE"]
        ].to_string(index=False)
    )

print("\nNORMALITY TEST SUMMARY")

if not normality_df.empty:

    print(
        normality_df[
            ["Target", "Shapiro_P_Value", "Interpretation"]
        ].to_string(index=False)
    )

print("\nLarge-error observations:", len(outlier_df))

print("\nReports saved in:")
print(OUTPUT_DIR.resolve())

print("\nGraphs saved in:")
print(GRAPH_DIR.resolve())

print("\nIMPORTANT:")
print(
    "This analysis uses out-of-fold predictions. "
    "Normality and IQR findings are diagnostic indicators, "
    "not proof of model validity or invalidity."
)
print(
    "Actual and generated records cannot be separated "
    "without a reliable row-level source label."
)
