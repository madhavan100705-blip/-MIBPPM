from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "caos_surface_confirmed.csv"
OUT = ROOT / "results" / "x50"

FEATURES = [
    "S_over_B", "H_over_B", "B_over_D",
    "T_over_B", "Pf_kg_m3", "XB_m", "E_GPa"
]
TARGET = "x50_m"

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

# Validate the dataset
required = FEATURES + [TARGET, "site", "blast_id"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

if df[required].isnull().any().any():
    raise ValueError("Missing values detected. Inspect the CSV before training.")

if not np.isfinite(df[FEATURES + [TARGET]].to_numpy(dtype=float)).all():
    raise ValueError("Non-finite numerical values detected.")

if (df[TARGET] <= 0).any():
    raise ValueError("X50 must be positive.")

X = df[FEATURES]
y = df[TARGET]
groups = df["site"]

models = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=42,
        n_jobs=1
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
    )
}

# Leave one entire mine/site out at a time
logo = LeaveOneGroupOut()
prediction_rows = []

print("\nMIBPPM - X50 Surface Blasting Experiment")
print("=" * 55)
print("Records:", len(df))
print("Sites:", groups.nunique())
print("Features:", len(FEATURES))
print("\nRunning leave-one-site-out validation...\n")

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups), start=1
):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    test_df = df.iloc[test_idx]

    held_site = test_df["site"].iloc[0]
    print(f"Fold {fold}: holding out {held_site}")

    for model_name, prototype in models.items():
        model = prototype
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        for i, pred in enumerate(predictions):
            prediction_rows.append({
                "blast_id": test_df.iloc[i]["blast_id"],
                "site": held_site,
                "model": model_name,
                "actual_x50_m": float(y.iloc[test_idx[i]]),
                "predicted_x50_m": float(pred),
                "error_m": float(pred - y.iloc[test_idx[i]])
            })

pred_df = pd.DataFrame(prediction_rows)
pred_df.to_csv(OUT / "site_holdout_predictions.csv", index=False)

# Aggregate out-of-fold metrics
metrics = []

for model_name in models:
    part = pred_df[pred_df["model"] == model_name]
    actual = part["actual_x50_m"]
    predicted = part["predicted_x50_m"]

    metrics.append({
        "model": model_name,
        "records": len(part),
        "MAE_m": mean_absolute_error(actual, predicted),
        "RMSE_m": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted)
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(OUT / "model_metrics.csv", index=False)

print("\nOVERALL OUT-OF-SITE RESULTS")
print(metrics_df.to_string(index=False))

# Per-site results
site_metrics = []
for (model_name, site), part in pred_df.groupby(["model", "site"]):
    site_metrics.append({
        "model": model_name,
        "site": site,
        "records": len(part),
        "MAE_m": mean_absolute_error(
            part["actual_x50_m"], part["predicted_x50_m"]
        )
    })

pd.DataFrame(site_metrics).to_csv(
    OUT / "per_site_metrics.csv", index=False
)

# Actual vs predicted plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, model_name in zip(axes, models):
    part = pred_df[pred_df["model"] == model_name]
    actual = part["actual_x50_m"]
    predicted = part["predicted_x50_m"]

    ax.scatter(actual, predicted, alpha=0.75)
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    ax.plot([low, high], [low, high], linestyle="--")

    ax.set_title(model_name)
    ax.set_xlabel("Actual X50 (m)")
    ax.set_ylabel("Predicted X50 (m)")

fig.suptitle("X50 Prediction - Leave-One-Site-Out Validation")
fig.tight_layout()
fig.savefig(OUT / "actual_vs_predicted.png", dpi=180)
plt.close(fig)

# Dataset diagnostics: repeated measurements are reported, not deleted.
duplicate_count = int(
    df.duplicated(subset=FEATURES + [TARGET], keep=False).sum()
)

summary = {
    "dataset": str(DATA),
    "records": len(df),
    "sites": int(groups.nunique()),
    "features": FEATURES,
    "target": TARGET,
    "repeated_feature_target_rows": duplicate_count,
    "validation": "LeaveOneGroupOut by site",
    "warning": (
        "Research benchmark only. Small dataset; results are not "
        "proof of generalisation to unseen mines."
    )
}

with open(OUT / "experiment_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nRepeated feature/target rows:", duplicate_count)
print("Results saved in:", OUT)
print("\nExperiment completed.")
