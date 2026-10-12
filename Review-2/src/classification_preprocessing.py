"""
Common preprocessing utilities for the Review-1 Adult Income classification track.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "AdultIncome.csv"


def load_adult_dataset(path=DATA_PATH):
    """Load Adult Income data from CSV; download through OpenML if absent."""
    path = Path(path)

    if path.exists():
        df = pd.read_csv(path)
    else:
        from sklearn.datasets import fetch_openml

        adult = fetch_openml(name="adult", version=2, as_frame=True)
        df = adult.frame.copy()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    df.columns = [str(c).strip() for c in df.columns]

    # OpenML/UCI variants can contain whitespace and a trailing period in labels.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"?": np.nan, "nan": np.nan})

    # Standardize the target column name.
    target_candidates = [c for c in df.columns if c.lower() in {"class", "income", "income-class"}]
    if target_candidates:
        target = target_candidates[0]
    else:
        target = df.columns[-1]

    df = df.rename(columns={target: "income"})
    df["income"] = (
        df["income"]
        .astype(str)
        .str.strip()
        .str.rstrip(".")
        .replace({"nan": np.nan})
    )

    # Remove rows with missing target, if any.
    df = df.dropna(subset=["income"]).reset_index(drop=True)

    return df


def prepare_features_target(df):
    """Separate predictors and target and normalize target labels."""
    data = df.copy()

    y = (
        data["income"]
        .astype(str)
        .str.strip()
        .str.rstrip(".")
        .map({"<=50K": 0, ">50K": 1})
    )

    valid = y.notna()
    data = data.loc[valid].copy()
    y = y.loc[valid].astype(int)

    X = data.drop(columns=["income"])

    # fnlwgt is a sampling weight rather than a descriptive house-style feature.
    # It is retained here because it is part of the standard Adult benchmark.
    return X, y


def build_preprocessor(X):
    """Create a leakage-safe preprocessing pipeline."""
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def save_result(result, filename):
    """Save one model's metrics as a CSV row."""

    from pathlib import Path

    # Review-1/results
    results_dir = Path(__file__).resolve().parent.parent / "results"

    output = results_dir / filename
    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([result]).to_csv(output, index=False)

    return output
