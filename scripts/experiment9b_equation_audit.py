
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

EXCEL = Path(
    r"C:\Users\madhavan.s\Downloads\SBPI.xlsx"
)

OUT = BASE / "results" / "sbpi" / "experiment9b"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_excel(
    EXCEL,
    sheet_name="Production Blast Dataset "
)

df.columns = df.columns.astype(str).str.strip()

# Inputs actually present in the workbook.
available = set(df.columns)

requirements = {
    "P80 - Kuznetsov": [
        "Burden (m)",
        "Spacing (m)",
        "Bench height (m)",
        "Charge mass (kg)",
        "Rock factor",
        "Explosive RWS"
    ],

    "P80 - Rosin-Rammler": [
        "X50 (m)",
        "Uniformity index"
    ],

    "PPV - USBM": [
        "Maximum charge per delay (kg)",
        "Monitoring distance (m)",
        "Site constant K",
        "Site exponent beta"
    ],

    "Flyrock - Ghasemi": [
        "Powder Factor (kg/m³)",
        "Stemming (m)",
        "Burden (m)",
        "Original equation parameters"
    ],

    "Crusher Productivity": [
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

# Parameters whose columns are not in Excel but may be
# available in external published records.
# We do not mark them available without evidence.
aliases = {
    "Burden (m)": "Burden (m)",
    "Spacing (m)": "Spacing (m)",
    "Powder Factor (kg/m³)": "Powder Factor (kg/m³)"
}

rows = []

for equation, fields in requirements.items():

    print("\n" + equation)
    print("-" * 45)

    for field in fields:

        column = aliases.get(field, field)

        status = (
            "AVAILABLE_IN_EXCEL"
            if column in available
            else "NOT_AVAILABLE_IN_EXCEL"
        )

        print(f"{field}: {status}")

        rows.append({
            "Equation": equation,
            "Required_parameter": field,
            "Status": status
        })

audit = pd.DataFrame(rows)

audit.to_csv(
    OUT / "equation_readiness.csv",
    index=False
)

# Preserve the actual column names and data statistics.
df.describe().T.to_csv(
    OUT / "dataset_statistics.csv"
)

print("\n" + "=" * 55)
print("EQUATION READINESS AUDIT COMPLETED")
print("=" * 55)

print("\nExcel columns:")
for col in df.columns:
    print("-", col)

print("\nSaved in:")
print(OUT.resolve())
