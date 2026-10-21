"""
Shared Preprocessing Pipeline for Sentinel Academic Fraud Detection System.
Team: Aniket Dubey, Raju Kumar

Ensures zero data leakage between training and runtime inference.
Single source of truth for feature engineering across ML models and API scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERICAL_FEATURES = [
    "amount",
    "amount_ratio",
    "hour",
    "is_night",
    "location_change",
    "device_change",
    "velocity_10m",
    "velocity_1h",
    "merchant_risk"
]

CATEGORICAL_FEATURES = [
    "channel"
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Fallback merchant risk values for scoring API
MERCHANT_RISK_MAP = {
    "grocery": 0.02,
    "food": 0.03,
    "retail": 0.05,
    "travel": 0.20,
    "electronics": 0.25,
    "crypto": 0.65,
    "gambling": 0.70
}


def create_preprocessor() -> ColumnTransformer:
    """
    Constructs a scikit-learn ColumnTransformer with scaling and encoding.
    Ensures safe handling of unknown categories and reproducible scaling.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def extract_features_df(data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
    """
    Extracts and validates features from either a DataFrame or a raw dictionary.
    Computes necessary ratios and default values if missing.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # Ensure amount_ratio exists
    if "amount_ratio" not in df.columns:
        if "customer_avg_amount" in df.columns:
            df["amount_ratio"] = df["amount"] / df["customer_avg_amount"].replace(0, 1.0)
        else:
            # Baseline assumption if customer average is not provided
            df["amount_ratio"] = df["amount"] / 2500.0

    # Ensure merchant_risk exists
    if "merchant_risk" not in df.columns:
        if "merchant_category" in df.columns:
            df["merchant_risk"] = df["merchant_category"].map(MERCHANT_RISK_MAP).fillna(0.05)
        else:
            df["merchant_risk"] = 0.05

    # Ensure hour and is_night exist
    if "hour" not in df.columns:
        df["hour"] = 12
    if "is_night" not in df.columns:
        df["is_night"] = df["hour"].apply(lambda h: 1 if 0 <= h <= 5 else 0)

    # Defaults for binary/velocity indicators
    for col, default_val in [
        ("location_change", 0),
        ("device_change", 0),
        ("velocity_10m", 1),
        ("velocity_1h", 1),
        ("channel", "WEB")
    ]:
        if col not in df.columns:
            df[col] = default_val

    # Ensure type consistency
    df["is_night"] = df["is_night"].astype(int)
    df["location_change"] = df["location_change"].astype(int)
    df["device_change"] = df["device_change"].astype(int)
    df["velocity_10m"] = df["velocity_10m"].astype(int)
    df["velocity_1h"] = df["velocity_1h"].astype(int)
    df["amount"] = df["amount"].astype(float)
    df["amount_ratio"] = df["amount_ratio"].astype(float)
    df["merchant_risk"] = df["merchant_risk"].astype(float)
    df["hour"] = df["hour"].astype(int)

    return df[ALL_FEATURE_COLUMNS]
