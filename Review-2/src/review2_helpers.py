"""Shared, leakage-safe utilities for Review 2 classification notebooks."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADULT_PATH = PROJECT_ROOT / "data" / "AdultIncome.csv"
RANDOM_STATE = 42

def load_adult_dataset(path=ADULT_PATH):
    """Load Adult Income data and normalize labels/missing markers."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(object).map(lambda value: value.strip() if isinstance(value, str) else value).replace({"?": np.nan, "": np.nan})
    if "income" not in df.columns:
        candidates = [c for c in df.columns if c.lower() in {"class", "income-class"}]
        df = df.rename(columns={candidates[0] if candidates else df.columns[-1]: "income"})
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    df = df.dropna(subset=["income"]).reset_index(drop=True)
    return df

def split_adult(df):
    """Return X/y and a consistent stratified 80/20 split."""
    y = df["income"].astype(str).str.strip().str.rstrip(".").map({"<=50K": 0, ">50K": 1})
    valid = y.notna()
    X = df.loc[valid].drop(columns=["income"]).copy()
    y = y.loc[valid].astype(int)
    return train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

def build_preprocessor(X):
    """Impute and encode within the pipeline; scaler/encoder fit only on training data."""
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                         ("scaler", StandardScaler())])
    return ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)], remainder="drop")
