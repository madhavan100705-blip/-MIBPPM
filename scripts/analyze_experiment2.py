import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/caos_surface_confirmed.csv"
PRED = ROOT / "results/x50/experiment2/predictions.csv"
OUT = ROOT / "results/x50/experiment3"

OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "S_over_B", "H_over_B", "B_over_D",
    "T_over_B", "Pf_kg_m3", "XB_m", "E_GPa"
]

TARGET = "x50_m"

df = pd.read_csv(DATA)
pred = pd.read_csv(PRED)

print("\nMIBPPM - EXPERIMENT 3: MODEL DIAGNOSTICS")
print("=" * 60)

# 1. Residual analysis
linear = pred[pred["model"] == "Linear Regression"].copy()
linear["residual_m"] = (
    linear["predicted_x50_m"] - linear["actual_x50_m"]
)
linear["absolute_error_m"] = linear["residual_m"].abs()

linear.to_csv(OUT / "linear_residuals.csv", index=False)

print("\n1. OVERALL RESIDUAL STATISTICS")
print("Mean residual:", round(linear["residual_m"].mean(), 5))
print("MAE:", round(linear["absolute_error_m"].mean(), 5))
print("RMSE:", round(np.sqrt(np.mean(linear["residual_m"] ** 2)), 5))

# 2. Site-wise residual analysis
site_results = linear.groupby("site").agg(
    records=("residual_m", "count"),
    mean_residual=("residual_m", "mean"),
    MAE=("absolute_error_m", "mean"),
    RMSE=("residual_m", lambda x: np.sqrt(np.mean(x ** 2)))
).reset_index()

site_results.to_csv(OUT / "site_residual_analysis.csv", index=False)

print("\n2. SITE-WISE RESIDUALS")
print(site_results.round(4).to_string(index=False))

# 3. Fit standardized linear regression in each training fold
X = df[FEATURES].astype(float)
y = df[TARGET].astype(float)
groups = df["site"].astype(str)

logo = LeaveOneGroupOut()
coefficient_rows = []

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups), start=1
):
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    regression = model.named_steps["linearregression"]

    row = {
        "held_out_site": groups.iloc[test_idx[0]],
        "intercept": regression.intercept_
    }

    for feature, value in zip(FEATURES, regression.coef_):
        row[feature] = value

    coefficient_rows.append(row)

coefficients = pd.DataFrame(coefficient_rows)
coefficients.to_csv(OUT / "fold_coefficients.csv", index=False)

print("\n3. STANDARDIZED COEFFICIENTS BY FOLD")
print(coefficients.round(4).to_string(index=False))

print("\n4. COEFFICIENT STABILITY")
coefficient_summary = coefficients[FEATURES].agg(
    ["mean", "std", "min", "max"]
).T

coefficient_summary.to_csv(OUT / "coefficient_stability.csv")
print(coefficient_summary.round(4).to_string())

# 4. Duplicate sensitivity experiment
# Primary dataset is never modified.
sensitivity_rows = []

for label, data in [
    ("All records", df.copy()),
    ("En2 excluded - sensitivity only",
     df[df["blast_id"] != "En2"].copy())
]:
    X_s = data[FEATURES].astype(float)
    y_s = data[TARGET].astype(float)
    groups_s = data["site"].astype(str)

    actual_all = []
    predicted_all = []

    for train_idx, test_idx in logo.split(X_s, y_s, groups_s):
        model = make_pipeline(StandardScaler(), LinearRegression())
        model.fit(X_s.iloc[train_idx], y_s.iloc[train_idx])

        predictions = model.predict(X_s.iloc[test_idx])

        actual_all.extend(y_s.iloc[test_idx].tolist())
        predicted_all.extend(predictions.tolist())

    sensitivity_rows.append({
        "dataset": label,
        "records": len(data),
        "MAE_m": mean_absolute_error(actual_all, predicted_all),
        "RMSE_m": np.sqrt(
            mean_squared_error(actual_all, predicted_all)
        ),
        "R2": r2_score(actual_all, predicted_all)
    })

sensitivity = pd.DataFrame(sensitivity_rows)
sensitivity.to_csv(OUT / "duplicate_sensitivity.csv", index=False)

print("\n5. DUPLICATE SENSITIVITY")
print(sensitivity.round(4).to_string(index=False))

# 5. Diagnostic graphs
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(
    linear["actual_x50_m"],
    linear["residual_m"]
)
axes[0].axhline(0, linestyle="--")
axes[0].set_xlabel("Actual X50 (m)")
axes[0].set_ylabel("Residual (Predicted - Actual) (m)")
axes[0].set_title("Residual vs Actual X50")
axes[0].grid(True, alpha=0.3)

axes[1].bar(
    site_results["site"],
    site_results["MAE"]
)
axes[1].set_xlabel("Site")
axes[1].set_ylabel("MAE (m)")
axes[1].set_title("Linear Regression Error by Site")
axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "diagnostic_graphs.png", dpi=160)
plt.close()

print("\n6. RESULTS SAVED")
print(OUT)
print("\nExperiment 3 completed.")
