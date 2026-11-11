
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EXCEL = Path(r"C:\Users\madhavan.s\Downloads\SBPI.xlsx")
OUTPUT = BASE / "results" / "sbpi" / "experiment9"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(
    EXCEL,
    sheet_name="Production Blast Dataset "
)

df.columns = df.columns.astype(str).str.strip()

print("\nMIBPPM EXPERIMENT 9")
print("MATHEMATICAL MODEL READINESS AUDIT")
print("=" * 60)

print("\nDATASET INFORMATION")
print("Records:", len(df))
print("Blast IDs:", df["Blast ID"].nunique())

# Inputs required for each mathematical model.
requirements = {
    "P80 - Kuznetsov/Rosin-Rammler": [
        "Burden",
        "Spacing",
        "Bench height",
        "Charge mass",
        "Rock factor",
        "Uniformity index"
    ],
    "PPV - Scaled distance": [
        "Maximum charge per delay",
        "Monitoring distance",
        "Site constants K and beta"
    ],
    "Flyrock - Empirical models": [
        "Charge mass",
        "Burden",
        "Stemming",
        "Site calibration constants"
    ],
    "Crusher productivity": [
        "Crusher capacity",
        "Feed size",
        "Operating hours",
        "Equipment availability"
    ],
    "SBPI": [
        "Original SBPI equation",
        "Indicator weights",
        "Normalization method"
    ]
}

available = {
    "Blast ID",
    "UCS (MPa)",
    "RQD (%)",
    "Burden (m)",
    "Spacing (m)",
    "Powder Factor (kg/m³)",
    "P80 (mm)",
    "PPV (mm/s)",
    "Flyrock (m)",
    "Crusher Productivity (t/h)",
    "SBPI"
}

# Report available and missing fields.
audit_rows = []

for model, fields in requirements.items():

    print("\n" + model)
    print("-" * 50)

    for field in fields:

        status = (
            "AVAILABLE"
            if field in available
            else "MISSING"
        )

        print(f"{field}: {status}")

        audit_rows.append({
            "Mathematical_Model": model,
            "Required_Information": field,
            "Status": status
        })

audit = pd.DataFrame(audit_rows)

audit.to_csv(
    OUTPUT / "mathematical_readiness.csv",
    index=False
)

# Dataset statistical summary.
summary = df.describe(
    include="all"
).transpose()

summary.to_csv(
    OUTPUT / "dataset_statistics.csv"
)

# Correlation analysis.
numeric = df.select_dtypes(include=np.number)

correlations = numeric.corr()

correlations.to_csv(
    OUTPUT / "correlation_matrix.csv"
)

print("\n" + "=" * 60)
print("MATHEMATICAL READINESS SUMMARY")
print("=" * 60)

print(
    audit.groupby(
        ["Mathematical_Model", "Status"]
    ).size().unstack(fill_value=0)
)

print("\nReports saved to:")
print(OUTPUT.resolve())

print("\nNo missing physical parameters were invented.")
