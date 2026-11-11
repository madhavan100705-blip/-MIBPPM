
"""
MIBPPM - SBPI Excel Dataset
Train and compare AI models for five blasting outcomes.

Run:
python scripts/train_sbpi_models.py --input "C:\\Users\\madhavan.s\\Downloads\\SBPI.xlsx"

Validation:
5-fold GroupKFold using Blast ID.
All records from one Blast ID stay in the same fold.
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

GROUP_COL = "Blast ID"

FEATURES = [
    "UCS (MPa)",
    "RQD (%)",
    "Burden (m)",
    "Spacing (m)",
    "Powder Factor (kg/m³)",
]

TARGETS = [
    "P80 (mm)",
    "PPV (mm/s)",
    "Flyrock (m)",
    "Crusher Productivity (t/h)",
    "SBPI",
]


def get_models():

    models = {
        "Mean Baseline": DummyRegressor(strategy="mean"),

        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(C=10, epsilon=0.1))
        ]),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=2,
            random_state=42
        )
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        )

    except ImportError:
        print("XGBoost unavailable. Skipping XGBoost.")

    return models


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to SBPI Excel file"
    )

    parser.add_argument(
        "--output",
        default="results/sbpi"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = output_dir / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Excel file not found: {input_path}"
        )

    print("\nReading Excel workbook...")

    # Find the correct sheet even if its name has trailing spaces.
    excel = pd.ExcelFile(input_path)

    sheet_map = {
        str(name).strip(): name
        for name in excel.sheet_names
    }

    expected_sheet = "Production Blast Dataset"

    if expected_sheet not in sheet_map:
        raise ValueError(
            "Could not find Production Blast Dataset.\n"
            f"Available sheets: {excel.sheet_names}"
        )

    actual_sheet_name = sheet_map[expected_sheet]

    df = pd.read_excel(
        input_path,
        sheet_name=actual_sheet_name
    )

    df.columns = [str(c).strip() for c in df.columns]

    required = [GROUP_COL] + FEATURES + TARGETS

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in Excel file: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    data = df[required].copy()

    # Convert model columns to numeric values.
    for col in FEATURES + TARGETS:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    before = len(data)

    data = data.dropna(
        subset=required
    ).reset_index(drop=True)

    dropped = before - len(data)

    if data[GROUP_COL].nunique() < 5:
        raise ValueError(
            "Need at least five distinct Blast IDs."
        )

    X = data[FEATURES]
    groups = data[GROUP_COL]

    print("\nDATASET SUMMARY")
    print("-----------------------------")
    print("Total usable rows:", len(data))
    print("Distinct Blast IDs:", groups.nunique())
    print("Rows removed:", dropped)
    print("Input features:", len(FEATURES))
    print("Prediction targets:", len(TARGETS))

    print("\nRows per Blast ID:")
    print(data.groupby(GROUP_COL).size())

    # Save cleaned dataset.
    data.to_csv(
        output_dir / "cleaned_dataset.csv",
        index=False
    )

    audit = {
        "source_file": str(input_path.resolve()),
        "sheet": str(actual_sheet_name),
        "usable_rows": int(len(data)),
        "dropped_rows": int(dropped),
        "distinct_blast_ids": int(groups.nunique()),
        "features": FEATURES,
        "targets": TARGETS,
        "validation": "5-fold GroupKFold by Blast ID",
        "warning": (
            "Small dataset. Metrics are exploratory and "
            "do not prove performance at new mines."
        )
    }

    with open(
        output_dir / "dataset_audit.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(audit, f, indent=4)

    models = get_models()

    cv = GroupKFold(n_splits=5)

    all_metrics = []
    all_predictions = []
    best_models = {}

    # Train one model comparison for each target.
    for target in TARGETS:

        print(f"\n\nTRAINING TARGET: {target}")
        print("=" * 55)

        y = data[target].astype(float)

        target_results = []

        for model_name, estimator in models.items():

            oof = np.full(
                len(data),
                np.nan,
                dtype=float
            )

            # Grouped cross-validation.
            for train_idx, test_idx in cv.split(
                X, y, groups
            ):

                model = clone(estimator)

                model.fit(
                    X.iloc[train_idx],
                    y.iloc[train_idx]
                )

                predictions = model.predict(
                    X.iloc[test_idx]
                )

                oof[test_idx] = predictions

            if np.isnan(oof).any():
                raise RuntimeError(
                    f"Missing predictions for {target}"
                )

            mae = mean_absolute_error(y, oof)

            rmse = np.sqrt(
                mean_squared_error(y, oof)
            )

            r2 = r2_score(y, oof)

            all_metrics.append({
                "Target": target,
                "Model": model_name,
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2)
            })

            for i in range(len(data)):

                all_predictions.append({
                    "Blast ID": data.loc[i, GROUP_COL],
                    "Target": target,
                    "Model": model_name,
                    "Actual": float(y.iloc[i]),
                    "Predicted": float(oof[i]),
                    "Residual": float(oof[i] - y.iloc[i])
                })

            target_results.append(
                (mae, model_name, estimator)
            )

            print(
                f"{model_name:22s} "
                f"MAE={mae:.5f}  "
                f"RMSE={rmse:.5f}  "
                f"R2={r2:.5f}"
            )

        # Select the model with lowest grouped-CV MAE.
        target_results.sort(
            key=lambda item: item[0]
        )

        best_mae, best_name, best_estimator = (
            target_results[0]
        )

        # Refit selected model on all available records.
        final_model = clone(best_estimator)

        final_model.fit(X, y)

        safe_name = re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            target
        ).strip("_").lower()

        model_path = (
            model_dir /
            f"{safe_name}_best_model.joblib"
        )

        joblib.dump(
            final_model,
            model_path
        )

        best_models[target] = {
            "best_model": best_name,
            "grouped_cv_MAE": float(best_mae),
            "saved_file": str(model_path)
        }

        print(
            f"\nBest model for {target}: "
            f"{best_name}"
        )

    # Save comparison results.
    metrics_df = pd.DataFrame(all_metrics)

    metrics_df = metrics_df.sort_values(
        ["Target", "MAE"]
    )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    metrics_df.to_csv(
        output_dir / "model_metrics.csv",
        index=False
    )

    predictions_df.to_csv(
        output_dir / "oof_predictions.csv",
        index=False
    )

    with open(
        output_dir / "best_models.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(best_models, f, indent=4)

    print("\n\nFINAL MODEL COMPARISON")
    print("=" * 65)

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

    print("\n\nBEST MODEL FOR EACH TARGET")
    print("=" * 45)

    for target, result in best_models.items():

        print(
            f"{target}: "
            f"{result['best_model']} "
            f"(MAE={result['grouped_cv_MAE']:.5f})"
        )

    print("\nTRAINING COMPLETED.")

    print(
        "\nResults saved in:",
        output_dir.resolve()
    )

    print(
        "\nIMPORTANT: Use grouped-CV metrics as "
        "the performance estimates. Saved models "
        "were refitted on all rows for later inference."
    )


if __name__ == "__main__":
    main()
