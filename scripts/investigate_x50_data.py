import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "data" / "raw" / "caos_surface_confirmed.csv"

df = pd.read_csv(FILE)

features = [
    "S_over_B", "H_over_B", "B_over_D",
    "T_over_B", "Pf_kg_m3", "XB_m", "E_GPa"
]

target = "x50_m"

print("\nMIBPPM - DATA QUALITY INVESTIGATION")
print("=" * 55)

print("\n1. DATASET SHAPE")
print(df.shape)

print("\n2. SITE-WISE TARGET DISTRIBUTION")
print(
    df.groupby("site")[target]
    .agg(["count", "mean", "std", "min", "max"])
    .round(4)
)

print("\n3. FEATURE CORRELATION WITH X50")
print(
    df[features + [target]]
    .corr(numeric_only=True)[target]
    .sort_values(ascending=False)
    .round(4)
)

print("\n4. EXACT DUPLICATE FEATURE/TARGET RECORDS")
duplicates = df[df.duplicated(subset=features + [target], keep=False)]

if duplicates.empty:
    print("No repeated feature/target records found.")
else:
    print(duplicates[["blast_id", "site"] + features + [target]].to_string(index=False))

print("\n5. SITE-WISE FEATURE DISTRIBUTION")
for site, group in df.groupby("site"):
    print(f"\n{site} ({len(group)} records)")
    print(group[features + [target]].mean().round(4).to_string())

print("\n6. MISSING VALUES")
print(df[features + [target]].isnull().sum())

print("\nInvestigation completed.")
