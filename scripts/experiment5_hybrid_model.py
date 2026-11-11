
"""Experiment 5: Hybrid Mathematics + AI X50 prediction.

Run from the MIBPPM project root:
    python scripts/experiment5_hybrid_model.py

Validation:
    Leave-One-Site-Out (LOSO).

The hybrid residual models are trained only on the other sites
in each fold. No held-out-site measurements are used for training.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "caos_surface_confirmed.csv"
MATH_PATH = ROOT / "results" / "x50" / "experiment4" / "mathematical_predictions.csv"
COMPARISON_PATH = ROOT / "results" / "x50" / "experiment4" / "comparison_predictions.csv"

OUT = ROOT / "results" / "x50" / "experiment5"

FEATURES = [
    "S_over_B",
    "H_over_B",
    "B_over_D",
    "T_over_B",
    "Pf_kg_m3",
    "XB_m",
    "E_GPa",
]

MATH_COLUMN = "kuznetsov_2012_x50_m"
AI_COLUMN = "predicted_x50_m"
TARGET = "x50_m"


def calculate_metrics(actual, predicted):
    return {
        "n": int(len(actual)),
        "MAE_m": float(mean_absolute_error(actual, predicted)),
        "RMSE_m": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def main():

    OUT.mkdir(parents=True, exist_ok=True)

    for path in [DATA_PATH, MATH_PATH, COMPARISON_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # ---------------------------------------------------------
    # 1. Load and merge the existing datasets
    # ---------------------------------------------------------

    data = pd.read_csv(DATA_PATH)
    math_data = pd.read_csv(MATH_PATH)
    previous = pd.read_csv(COMPARISON_PATH)

    ai = previous[
        ["blast_id", "site", AI_COLUMN]
    ].copy()

    df = data.merge(
        math_data[["blast_id", "site", MATH_COLUMN]],
        on=["blast_id", "site"],
        how="inner",
        validate="one_to_one",
    )

    df = df.merge(
        ai,
        on=["blast_id", "site"],
        how="inner",
        validate="one_to_one",
    )

    df = df.dropna(
        subset=FEATURES + [TARGET, MATH_COLUMN, AI_COLUMN]
    ).copy()

    df = df.reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(
            f"Only {len(df)} complete records found. "
            "Check the input files."
        )

    if df["site"].nunique() < 2:
        raise ValueError("At least two sites are required.")

    # Confirm the existing AI and mathematical predictions
    # correspond to the same actual measurements.
    if not np.all(np.isfinite(df[FEATURES].to_numpy())):
        raise ValueError("Non-finite feature values detected.")

    print("\nEXPERIMENT 5 DATA CHECK")
    print("----------------------")
    print("Records:", len(df))
    print("Sites:", df["site"].nunique())
    print("Site names:", sorted(df["site"].unique()))

    # ---------------------------------------------------------
    # 2. Prepare models and output columns
    # ---------------------------------------------------------

    X_features = df[FEATURES].to_numpy(dtype=float)

    # The hybrid learns the error of the mathematical equation:
    # residual = actual X50 - mathematical X50.
    #
    # The mathematical prediction is also supplied as an input.
    X_hybrid = np.column_stack([
        X_features,
        df[MATH_COLUMN].to_numpy(dtype=float),
    ])

    y = df[TARGET].to_numpy(dtype=float)
    groups = df["site"].to_numpy()

    df["hybrid_ridge_prediction_m"] = np.nan
    df["hybrid_rf_prediction_m"] = np.nan
    df["ensemble_50_50_prediction_m"] = (
        0.5 * df[MATH_COLUMN] + 0.5 * df[AI_COLUMN]
    )

    df["ridge_predicted_correction_m"] = np.nan
    df["rf_predicted_correction_m"] = np.nan
    df["held_out_site"] = ""

    # ---------------------------------------------------------
    # 3. Leave-One-Site-Out hybrid validation
    # ---------------------------------------------------------

    logo = LeaveOneGroupOut()

    for fold_number, (train_idx, test_idx) in enumerate(
        logo.split(X_hybrid, y, groups),
        start=1,
    ):

        X_train = X_hybrid[train_idx]
        X_test = X_hybrid[test_idx]

        # Residual to learn:
        # actual X50 minus Kuznetsov X50.
        y_train_residual = (
            y[train_idx]
            - df.iloc[train_idx][MATH_COLUMN].to_numpy(dtype=float)
        )

        # Ridge residual correction.
        ridge_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ])

        ridge_model.fit(X_train, y_train_residual)

        ridge_correction = ridge_model.predict(X_test)

        math_test = df.iloc[test_idx][MATH_COLUMN].to_numpy(
            dtype=float
        )

        ridge_hybrid = math_test + ridge_correction

        # Random Forest residual correction.
        rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=3,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        )

        rf_model.fit(X_train, y_train_residual)

        rf_correction = rf_model.predict(X_test)

        rf_hybrid = math_test + rf_correction

        # Save predictions for the held-out site.
        df.loc[
            test_idx, "hybrid_ridge_prediction_m"
        ] = ridge_hybrid

        df.loc[
            test_idx, "hybrid_rf_prediction_m"
        ] = rf_hybrid

        df.loc[
            test_idx, "ridge_predicted_correction_m"
        ] = ridge_correction

        df.loc[
            test_idx, "rf_predicted_correction_m"
        ] = rf_correction

        held_site = str(groups[test_idx][0])

        df.loc[test_idx, "held_out_site"] = held_site

        print(
            f"Fold {fold_number}: held out {held_site} "
            f"({len(test_idx)} records)"
        )

    # Check that every record received an OOF hybrid prediction.
    for column in [
        "hybrid_ridge_prediction_m",
        "hybrid_rf_prediction_m",
    ]:
        if df[column].isna().any():
            raise ValueError(
                f"Missing out-of-fold predictions in {column}"
            )

    # ---------------------------------------------------------
    # 4. Calculate metrics on the same records
    # ---------------------------------------------------------

    model_columns = {
        "Kuznetsov 2012": MATH_COLUMN,
        "Linear Regression AI": AI_COLUMN,
        "Hybrid Ridge Correction": "hybrid_ridge_prediction_m",
        "Hybrid Random Forest Correction": "hybrid_rf_prediction_m",
        "50:50 Mathematics-AI Ensemble": "ensemble_50_50_prediction_m",
    }

    metric_rows = []

    for model_name, column in model_columns.items():

        values = calculate_metrics(
            df[TARGET].to_numpy(),
            df[column].to_numpy(),
        )

        values["model"] = model_name
        metric_rows.append(values)

    metrics = pd.DataFrame(metric_rows)[
        ["model", "n", "MAE_m", "RMSE_m", "R2"]
    ]

    metrics = metrics.sort_values(
        "MAE_m"
    ).reset_index(drop=True)

    metrics.to_csv(
        OUT / "model_metrics.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 5. Per-site metrics
    # ---------------------------------------------------------

    site_rows = []

    for site, site_data in df.groupby("site"):

        for model_name, column in model_columns.items():

            values = calculate_metrics(
                site_data[TARGET].to_numpy(),
                site_data[column].to_numpy(),
            )

            values["site"] = site
            values["model"] = model_name

            site_rows.append(values)

    site_metrics = pd.DataFrame(site_rows)

    site_metrics.to_csv(
        OUT / "per_site_metrics.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 6. Save prediction and residual details
    # ---------------------------------------------------------

    df.to_csv(
        OUT / "hybrid_predictions.csv",
        index=False,
    )

    residual_rows = []

    for model_name, column in model_columns.items():

        for record in df.to_dict(orient="records"):

            actual = float(record[TARGET])
            predicted = float(record[column])

            residual_rows.append({
                "blast_id": record["blast_id"],
                "site": record["site"],
                "model": model_name,
                "actual_x50_m": actual,
                "predicted_x50_m": predicted,
                "residual_m": predicted - actual,
                "absolute_error_m": abs(predicted - actual),
            })

    pd.DataFrame(residual_rows).to_csv(
        OUT / "residuals.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 7. Actual vs predicted graph
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 10),
    )

    axes = axes.flatten()

    max_value = max(
        float(df[TARGET].max()),
        max(float(df[column].max()) for column in model_columns.values()),
    ) * 1.05

    for ax, (model_name, column) in zip(
        axes,
        model_columns.items(),
    ):

        ax.scatter(
            df[TARGET],
            df[column],
            alpha=0.75,
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
        ax.set_title(model_name)
        ax.grid(True, alpha=0.25)

    axes[-1].axis("off")

    fig.suptitle(
        "Experiment 5: Hybrid Mathematics and AI",
        fontsize=15,
    )

    fig.tight_layout()

    fig.savefig(
        OUT / "actual_vs_predicted.png",
        dpi=180,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # 8. MAE comparison graph
    # ---------------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        metrics["model"],
        metrics["MAE_m"],
    )

    ax.set_ylabel("MAE (m)")
    ax.set_title("Experiment 5: Model MAE comparison")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUT / "mae_comparison.png",
        dpi=180,
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # 9. Experiment summary
    # ---------------------------------------------------------

    best = metrics.iloc[0]

    summary = {
        "experiment": "Experiment 5 - Hybrid Mathematics and AI",
        "records": int(len(df)),
        "sites": int(df["site"].nunique()),
        "validation": "Leave-One-Site-Out",
        "target": TARGET,
        "mathematical_baseline": (
            "Kuznetsov 2012-derived site factor"
        ),
        "ai_baseline": (
            "Existing Linear Regression LOSO out-of-fold predictions"
        ),
        "hybrid_method": (
            "AI predicts the residual between actual X50 and "
            "Kuznetsov mathematical X50."
        ),
        "hybrid_validation": (
            "Residual models are retrained inside each LOSO fold. "
            "The held-out site's measurements are not used for training."
        ),
        "ensemble_method": (
            "Fixed 50:50 average of mathematical and existing AI predictions."
        ),
        "best_model_by_mae": str(best["model"]),
        "metrics": metric_rows,
        "limitations": [
            "Only 85 records from eight sites are available.",
            "The mathematical rock factors were back-solved from published predictions.",
            "The existing AI predictions are out-of-fold predictions from Experiment 2.",
            "Results measure generalisation to held-out sites in this dataset, not all mines.",
            "The fixed 50:50 ensemble weight was not optimized.",
        ],
    }

    with open(
        OUT / "experiment_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    # ---------------------------------------------------------
    # 10. Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 65)
    print("EXPERIMENT 5 COMPLETE")
    print("=" * 65)

    print("\nMODEL COMPARISON:")

    print(
        metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nBEST MODEL BY MAE:")
    print(best["model"])
    print(f"MAE: {best['MAE_m']:.6f} m")

    print("\nOUTPUT FOLDER:")
    print(OUT)

    print("\nFILES CREATED:")

    for filename in [
        "model_metrics.csv",
        "per_site_metrics.csv",
        "hybrid_predictions.csv",
        "residuals.csv",
        "actual_vs_predicted.png",
        "mae_comparison.png",
        "experiment_summary.json",
    ]:
        print(" -", filename)


if __name__ == "__main__":
    main()
