from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

DEFAULT_DROP = {
    "blast_id", "mine_id", "source_id", "source_url", "event_date",
    "PPV_actual", "X50_actual", "X80_actual", "flyrock_actual",
    "backbreak_actual", "air_overpressure_actual", "cost_actual", "productivity_actual"
}

def train_regressor(df: pd.DataFrame, target: str, model_dir: str = "models") -> dict:
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")
    data = df.dropna(subset=[target]).copy()
    if len(data) < 12:
        raise ValueError(f"Need at least 12 measured rows for a prototype split; found {len(data)}.")
    y = pd.to_numeric(data[target], errors="coerce")
    data = data.loc[y.notna()].copy()
    y = y.loc[y.notna()]
    drop_cols = DEFAULT_DROP - {target}
    X = data.drop(columns=[c for c in drop_cols if c in data.columns] + [target], errors="ignore")
    # Exclude columns that are entirely empty.
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("No usable predictor columns remain.")
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                               ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                               ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    prep = ColumnTransformer(transformers)
    estimator = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                      random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", prep), ("model", estimator)])
    if "mine_id" in data.columns and data["mine_id"].nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=data["mine_id"]))
        split_method = "grouped by mine_id (single holdout)"
    else:
        train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
        split_method = "random holdout; not evidence of cross-mine generalization"
    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = pipe.predict(X.iloc[test_idx])
    actual = y.iloc[test_idx].to_numpy()
    metrics = {
        "n_total_measured": int(len(data)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "split_method": split_method,
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(mean_squared_error(actual, pred) ** 0.5),
        "R2": float(r2_score(actual, pred)) if len(actual) >= 2 else None,
        "bias_pred_minus_actual": float(np.mean(pred - actual)),
        "target": target,
        "model": "RandomForestRegressor"
    }
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "feature_columns": list(X.columns), "target": target},
                out / f"{target}_random_forest.joblib")
    pd.DataFrame({"actual": actual, "ai_prediction": pred}).to_csv(out / f"{target}_holdout_predictions.csv", index=False)
    (out / f"{target}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
