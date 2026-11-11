from pathlib import Path
import pandas as pd

REQUIRED_ID_COLUMNS = ["blast_id", "mine_id", "source_id"]

def load_blast_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_ID_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required provenance/identity columns: {missing}")
    if df["blast_id"].isna().any():
        raise ValueError("blast_id cannot be blank.")
    return df

def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "mines": int(df["mine_id"].nunique()),
        "sources": int(df["source_id"].nunique()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items()},
    }
