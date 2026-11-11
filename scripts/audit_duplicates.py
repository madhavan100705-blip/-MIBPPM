import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/caos_surface_confirmed.csv"
OUTPUT = ROOT / "results/x50/duplicate_audit.csv"

df = pd.read_csv(SOURCE)

features = [
    "S_over_B", "H_over_B", "B_over_D",
    "T_over_B", "Pf_kg_m3", "XB_m", "E_GPa", "x50_m"
]

duplicates = df[df.duplicated(subset=features, keep=False)].copy()

duplicates["duplicate_group"] = "DUP-" + (
    duplicates.groupby(features, dropna=False).ngroup() + 1
).astype(str).str.zfill(3)

duplicates["verification_status"] = "Unverified"
duplicates["verification_notes"] = (
    "Identical recorded features and target; check original paper "
    "to establish whether these are independent blasts."
)
duplicates["training_decision"] = "Retain pending verification"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
duplicates.to_csv(OUTPUT, index=False)

print("Duplicate audit updated.")
print(duplicates[[
    "blast_id", "duplicate_group",
    "verification_status", "training_decision"
]].to_string(index=False))
print("Saved:", OUTPUT)
