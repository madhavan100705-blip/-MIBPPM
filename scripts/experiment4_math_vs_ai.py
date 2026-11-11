
"""Experiment 4: Mathematical vs AI X50 prediction comparison.

Run from the MIBPPM project root:
    python scripts/experiment4_math_vs_ai.py

IMPORTANT:
The site rock factors are back-solved from published predictions.
This is a calibrated comparison, NOT independent validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[1]
CAOS_ROOT = ROOT.parent / "CAOS_BlastFrag-main"

if str(CAOS_ROOT) not in sys.path:
    sys.path.insert(0, str(CAOS_ROOT))

from blastfrag import load_training_corpus
from blastfrag.geometry import reconstruct_pattern
from blastfrag.classical import kuznetsov_x50_m
from blastfrag.rockfactor import (
    SITE_ROCK_FACTOR,
    SITE_ROCK_FACTOR_2010,
)

DATA_PATH = ROOT / "data" / "raw" / "caos_surface_confirmed.csv"
AI_PATH = ROOT / "results" / "x50" / "experiment2" / "predictions.csv"
OUT = ROOT / "results" / "x50" / "experiment4"


def calculate_metrics(actual, predicted):
    return {
        "n": int(len(actual)),
        "MAE_m": float(mean_absolute_error(actual, predicted)),
        "RMSE_m": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "R2": float(r2_score(actual, predicted))
        if len(actual) >= 2 else float("nan"),
    }


def main():

    OUT.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not AI_PATH.exists():
        raise FileNotFoundError(
            f"Experiment 2 predictions not found: {AI_PATH}"
        )

    # Load the selected surface dataset.
    source_df = pd.read_csv(DATA_PATH)

    # Load existing AI predictions.
    ai_all = pd.read_csv(AI_PATH)

    ai = ai_all.loc[
        ai_all["model"].eq("Linear Regression"),
        ["blast_id", "site", "actual_x50_m", "predicted_x50_m"],
    ].copy()

    if ai.empty:
        raise ValueError(
            "No predictions found for Linear Regression."
        )

    # Load original CAOS blast records.
    selected_ids = set(source_df["blast_id"].astype(str))

    blasts = {
        blast.blast_id: blast
        for blast in load_training_corpus()
        if blast.blast_id in selected_ids
    }

    mathematical_rows = []

    # Calculate mathematical predictions.
    for record in source_df.to_dict(orient="records"):

        blast_id = str(record["blast_id"])
        site = str(record["site"])
        actual_x50 = float(record["x50_m"])

        result = {
            "blast_id": blast_id,
            "site": site,
            "actual_x50_m": actual_x50,
            "rock_volume_m3": np.nan,
            "charge_mass_kg": np.nan,
            "geometry_status": "not_available",
            "kuznetsov_2012_x50_m": np.nan,
            "kuznetsov_2010_x50_m": np.nan,
            "kuznetsov_2012_status": "factor_unavailable",
            "kuznetsov_2010_status": "factor_unavailable",
            "caveat": "",
        }

        blast = blasts.get(blast_id)

        if blast is None:
            result["geometry_status"] = "blast_record_not_found"
            mathematical_rows.append(result)
            continue

        try:

            # Reconstruct blast geometry.
            pattern = reconstruct_pattern(blast)

            result["geometry_status"] = (
                "reconstructed_from_published_site_geometry"
            )

            result["rock_volume_m3"] = float(
                pattern.rock_volume_m3
            )

            result["charge_mass_kg"] = float(
                pattern.charge_mass_kg
            )

            if site == "Soma":
                result["caveat"] = (
                    "Source has unresolved burden/diameter conflict."
                )

            # Calculate with the two published-factor scenarios.
            scenarios = [
                (
                    SITE_ROCK_FACTOR,
                    "kuznetsov_2012_x50_m",
                    "kuznetsov_2012_status",
                    "2012",
                ),
                (
                    SITE_ROCK_FACTOR_2010,
                    "kuznetsov_2010_x50_m",
                    "kuznetsov_2010_status",
                    "2010",
                ),
            ]

            for factor_map, output_col, status_col, year in scenarios:

                factor = factor_map.get(site)

                if factor is None:
                    result[status_col] = "site_factor_unavailable"
                    continue

                prediction = kuznetsov_x50_m(
                    pattern,
                    factor,
                )

                result[output_col] = float(prediction)

                result[status_col] = (
                    f"calculated_using_{year}_back_solved_site_factor"
                )

        except Exception as error:

            result["geometry_status"] = (
                f"abstained: {type(error).__name__}: {error}"
            )

            result["kuznetsov_2012_status"] = "geometry_unavailable"
            result["kuznetsov_2010_status"] = "geometry_unavailable"

        mathematical_rows.append(result)

    math_df = pd.DataFrame(mathematical_rows)

    math_df.to_csv(
        OUT / "mathematical_predictions.csv",
        index=False,
    )

    # Merge mathematical and AI predictions.
    merged = math_df.merge(
        ai,
        on=["blast_id", "site"],
        how="left",
        suffixes=("", "_ai"),
    )

    # Verify actual values match.
    valid_ai = merged["actual_x50_m_ai"].notna()

    if valid_ai.any():

        if not np.allclose(
            merged.loc[valid_ai, "actual_x50_m"],
            merged.loc[valid_ai, "actual_x50_m_ai"],
            atol=1e-10,
        ):
            raise ValueError(
                "Actual X50 values differ between datasets."
            )

    merged.drop(
        columns=["actual_x50_m_ai"],
        inplace=True,
    )

    merged.to_csv(
        OUT / "comparison_predictions.csv",
        index=False,
    )

    # Compare all three prediction approaches.
    model_specs = [
        (
            "Kuznetsov 2012-derived factor",
            "kuznetsov_2012_x50_m",
        ),
        (
            "Kuznetsov 2010-derived factor",
            "kuznetsov_2010_x50_m",
        ),
        (
            "Linear Regression (LOSO OOF)",
            "predicted_x50_m",
        ),
    ]

    metric_rows = []
    site_rows = []

    for model_name, prediction_column in model_specs:

        valid = merged[prediction_column].notna()
        subset = merged.loc[valid].copy()

        if subset.empty:
            continue

        model_metrics = calculate_metrics(
            subset["actual_x50_m"],
            subset[prediction_column],
        )

        model_metrics.update({
            "model": model_name,
            "coverage_n": int(valid.sum()),
            "dataset_n": int(len(merged)),
        })

        metric_rows.append(model_metrics)

        for site, group in subset.groupby("site"):

            site_metrics = calculate_metrics(
                group["actual_x50_m"],
                group[prediction_column],
            )

            site_metrics.update({
                "model": model_name,
                "site": site,
            })

            site_rows.append(site_metrics)

    metrics_df = pd.DataFrame(metric_rows)

    metrics_df = metrics_df[
        [
            "model",
            "n",
            "coverage_n",
            "dataset_n",
            "MAE_m",
            "RMSE_m",
            "R2",
        ]
    ]

    site_df = pd.DataFrame(site_rows)

    site_df.to_csv(
        OUT / "per_site_metrics.csv",
        index=False,
    )

    metrics_df.to_csv(
        OUT / "model_metrics.csv",
        index=False,
    )

    # Create residual data.
    residual_rows = []

    for model_name, prediction_column in model_specs:

        valid = merged[prediction_column].notna()

        for record in merged.loc[valid].to_dict(orient="records"):

            predicted = record[prediction_column]
            actual = record["actual_x50_m"]

            residual_rows.append({
                "blast_id": record["blast_id"],
                "site": record["site"],
                "model": model_name,
                "actual_x50_m": actual,
                "predicted_x50_m": predicted,
                "residual_pred_minus_actual_m": predicted - actual,
                "absolute_error_m": abs(predicted - actual),
            })

    pd.DataFrame(residual_rows).to_csv(
        OUT / "residuals.csv",
        index=False,
    )

    # Actual vs predicted graphs.
    plot_models = [
        (name, column)
        for name, column in model_specs
        if merged[column].notna().any()
    ]

    fig, axes = plt.subplots(
        1,
        len(plot_models),
        figsize=(6 * len(plot_models), 5),
        squeeze=False,
    )

    max_value = max(
        float(merged["actual_x50_m"].max()),
        max(
            float(merged[column].max())
            for _, column in plot_models
        ),
    ) * 1.05

    for ax, (name, column) in zip(axes[0], plot_models):

        valid = merged[column].notna()

        ax.scatter(
            merged.loc[valid, "actual_x50_m"],
            merged.loc[valid, column],
            alpha=0.8,
        )

        ax.plot(
            [0, max_value],
            [0, max_value],
            linestyle="--",
        )

        ax.set_xlim(0, max_value)
        ax.set_ylim(0, max_value)

        ax.set_xlabel("Actual X50 (m)")
        ax.set_ylabel("Predicted X50 (m)")
        ax.set_title(name)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUT / "actual_vs_predicted.png",
        dpi=180,
    )

    plt.close(fig)

    # MAE comparison graph.
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        metrics_df["model"],
        metrics_df["MAE_m"],
    )

    ax.set_ylabel("MAE (m)")
    ax.set_title("Experiment 4: X50 MAE comparison")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUT / "mae_comparison.png",
        dpi=180,
    )

    plt.close(fig)

    # Save experiment summary.
    summary = {
        "experiment": "Experiment 4: mathematical vs AI X50",
        "records": int(len(merged)),
        "sites": sorted(
            merged["site"].unique().tolist()
        ),
        "ai_model": "Linear Regression",
        "ai_validation": (
            "Existing leave-one-site-out out-of-fold predictions "
            "from Experiment 2."
        ),
        "mathematical_method": (
            "Kuznetsov X50 using reconstructed pattern geometry "
            "and site-specific back-solved rock factors."
        ),
        "factor_caveat": (
            "2010 and 2012 rock factors are back-solved from "
            "published predictions, not independently measured."
        ),
        "geometry_caveats": [
            "Soma has an unresolved burden/diameter conflict.",
            "Miami is not included in the selected surface dataset.",
            "2010-derived factor is unavailable for Reocin."
        ],
        "metrics": metric_rows,
        "interpretation_warning": (
            "Prediction coverage differs between methods. "
            "Compare coverage before interpreting the metrics. "
            "This is not independent validation of Kuznetsov."
        ),
    }

    with open(
        OUT / "experiment_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    # Print results.
    print("\nEXPERIMENT 4 COMPLETE")

    print(f"\nOutput folder:\n{OUT}")

    print("\nMODEL METRICS:")

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nMATHEMATICAL PREDICTION COVERAGE:")

    for column in [
        "kuznetsov_2012_status",
        "kuznetsov_2010_status",
    ]:

        print(
            column,
            math_df[column].value_counts(dropna=False).to_dict(),
        )

    print("\nFiles created:")

    for filename in [
        "mathematical_predictions.csv",
        "comparison_predictions.csv",
        "model_metrics.csv",
        "per_site_metrics.csv",
        "residuals.csv",
        "actual_vs_predicted.png",
        "mae_comparison.png",
        "experiment_summary.json",
    ]:

        print(" -", OUT / filename)


if __name__ == "__main__":
    main()
