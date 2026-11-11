import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/caos_surface_confirmed.csv"
OUT = ROOT / "results/x50/experiment2"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "S_over_B", "H_over_B", "B_over_D",
    "T_over_B", "Pf_kg_m3", "XB_m", "E_GPa"
]
TARGET = "x50_m"

df = pd.read_csv(DATA)

X = df[FEATURES].astype(float)
y = df[TARGET].astype(float)
groups = df["site"].astype(str)

if X.isnull().any().any() or y.isnull().any():
    raise ValueError("Missing values found. Resolve them before training.")

if (y <= 0).any():
    raise ValueError("X50 must be positive for log-target experiments.")

models = {
    "Mean baseline": DummyRegressor(strategy="mean"),

    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),

    "Linear Regression - Log X50": TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            LinearRegression()
        ),
        func=np.log,
        inverse_func=np.exp
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=42,
        n_jobs=1
    ),

    "Random Forest - Log X50": TransformedTargetRegressor(
        regressor=RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=42,
            n_jobs=1
        ),
        func=np.log,
        inverse_func=np.exp
    ),

    "XGBoost": XGBRegressor(
        n_estimators=150,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        verbosity=0
    ),

    "XGBoost - Log X50": TransformedTargetRegressor(
        regressor=XGBRegressor(
            n_estimators=150,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
            verbosity=0
        ),
        func=np.log,
        inverse_func=np.exp
    )
}

logo = LeaveOneGroupOut()
prediction_rows = []

print("\nMIBPPM - EXPERIMENT 2")
print("=" * 60)
print("Records:", len(df))
print("Sites:", groups.nunique())
print("Features:", len(FEATURES))
print("Validation: Leave-one-site-out")
print("\nRunning experiments...\n")

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups), start=1
):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    test_sites = groups.iloc[test_idx].unique()

    site = test_sites[0]
    print(f"Fold {fold}: holding out {site}")

    for model_name, estimator in models.items():
        fitted = clone(estimator)
        fitted.fit(X_train, y_train)
        predictions = fitted.predict(X_test)

        for i, prediction in enumerate(predictions):
            prediction_rows.append({
                "blast_id": df.iloc[test_idx[i]]["blast_id"],
                "site": site,
                "model": model_name,
                "actual_x50_m": float(y_test.iloc[i]),
                "predicted_x50_m": float(prediction),
                "error_m": float(prediction - y_test.iloc[i])
            })

pred = pd.DataFrame(prediction_rows)
pred.to_csv(OUT / "predictions.csv", index=False)

metrics = []

for model_name, part in pred.groupby("model", sort=False):
    actual = part["actual_x50_m"]
    predicted = part["predicted_x50_m"]

    metrics.append({
        "model": model_name,
        "records": len(part),
        "MAE_m": mean_absolute_error(actual, predicted),
        "RMSE_m": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    })

metrics_df = pd.DataFrame(metrics).sort_values("MAE_m")
metrics_df.to_csv(OUT / "model_metrics.csv", index=False)

site_metrics = []

for (model_name, site), part in pred.groupby(["model", "site"]):
    site_metrics.append({
        "model": model_name,
        "site": site,
        "records": len(part),
        "MAE_m": mean_absolute_error(
            part["actual_x50_m"], part["predicted_x50_m"]
        ),
        "RMSE_m": np.sqrt(mean_squared_error(
            part["actual_x50_m"], part["predicted_x50_m"]
        ))
    })

pd.DataFrame(site_metrics).to_csv(
    OUT / "per_site_metrics.csv", index=False
)

print("\nEXPERIMENT 2 OVERALL RESULTS")
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Actual vs predicted plots for each model
model_names = metrics_df["model"].tolist()
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for ax, name in zip(axes, model_names):
    part = pred[pred["model"] == name]
    ax.scatter(part["actual_x50_m"], part["predicted_x50_m"])
    low = min(part["actual_x50_m"].min(), part["predicted_x50_m"].min())
    high = max(part["actual_x50_m"].max(), part["predicted_x50_m"].max())
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_title(name)
    ax.set_xlabel("Actual X50 (m)")
    ax.set_ylabel("Predicted X50 (m)")
    ax.grid(True, alpha=0.3)

for ax in axes[len(model_names):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=160)
plt.close()

summary = {
    "experiment": "Experiment 2",
    "records": int(len(df)),
    "sites": int(groups.nunique()),
    "features": FEATURES,
    "target": TARGET,
    "validation": "LeaveOneGroupOut by site",
    "models": model_names,
    "duplicate_policy": "No records removed",
    "warning": (
        "Small research benchmark. Site-wise validation estimates "
        "performance on unseen sites represented by this dataset; "
        "it does not prove generalisation to all mines."
    )
}

with open(OUT / "experiment_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nFiles saved in:", OUT)
print("Experiment 2 completed.")
