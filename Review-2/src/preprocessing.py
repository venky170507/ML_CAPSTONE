
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "SalePrice"

BASE_DROP = ["SalePrice", "Order", "PID"]

POLY_NUMERIC_FEATURES = [
    "Overall Qual", "Gr Liv Area", "Total Bsmt SF", "Garage Cars",
    "Year Built", "Year Remod/Add", "1st Flr SF", "Full Bath",
    "TotRms AbvGrd", "Garage Area"
]

def load_and_engineer(csv_path):
    df = pd.read_csv(csv_path).copy()
    # Engineered features use only predictors and therefore do not leak the target.
    df["TotalSF"] = df[["Total Bsmt SF", "1st Flr SF", "2nd Flr SF"]].fillna(0).sum(axis=1)
    df["TotalBathrooms"] = (
        df["Full Bath"].fillna(0)
        + 0.5 * df["Half Bath"].fillna(0)
        + df["Bsmt Full Bath"].fillna(0)
        + 0.5 * df["Bsmt Half Bath"].fillna(0)
    )
    X = df.drop(columns=BASE_DROP)
    if "MS SubClass" in X.columns:
        X["MS SubClass"] = X["MS SubClass"].astype(str)
    y = df[TARGET].copy()
    return df, X, y

def split_data(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

def remove_training_outliers(X_train, y_train, feature="Gr Liv Area", multiplier=3.0):
    # Threshold is learned from TRAINING data only; test data remains untouched.
    q1 = X_train[feature].quantile(0.25)
    q3 = X_train[feature].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + multiplier * iqr
    mask = X_train[feature].fillna(X_train[feature].median()) <= upper
    return X_train.loc[mask].copy(), y_train.loc[mask].copy(), upper

def column_lists(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical

def make_preprocessor(X, scale_numeric=True):
    numeric, categorical = column_lists(X)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", Pipeline(categorical_steps), categorical),
        ],
        remainder="drop"
    )

def make_polynomial_preprocessor(X, degree=2):
    _, categorical = column_lists(X)
    poly_numeric = [c for c in POLY_NUMERIC_FEATURES if c in X.columns]
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler())
    ])
    return ColumnTransformer([
        ("poly_num", num_pipe, poly_numeric),
        ("cat", cat_pipe, categorical),
    ])

def regression_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }
