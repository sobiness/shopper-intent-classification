"""Loading, target coercion and the stratified train/test split."""

# Keeps the builtin-generic annotations below importable on Python 3.8, which
# some lab environments still run.
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    FEATURES,
    RANDOM_STATE,
    RAW_DATA_FILE,
    TARGET,
    TEST_DATA_FILE,
    TEST_SIZE,
)

# Accepted spellings of the target, so an uploaded CSV is not rejected over
# formatting. The source file ships booleans; Excel round-trips often do not.
_TRUE_TOKENS = {"true", "1", "yes", "y", "purchase", "t"}
_FALSE_TOKENS = {"false", "0", "no", "n", "no purchase", "f"}


def load_raw() -> pd.DataFrame:
    """Read the full UCI file as downloaded."""
    return pd.read_csv(RAW_DATA_FILE)


def coerce_target(series: pd.Series) -> pd.Series:
    """Normalise the target to integers 0/1, whatever form it arrived in."""
    if series.dtype == bool:
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    normalised = series.astype(str).str.strip().str.lower()
    mapped = normalised.map(
        lambda token: 1
        if token in _TRUE_TOKENS
        else (0 if token in _FALSE_TOKENS else np.nan)
    )
    if mapped.isna().any():
        unexpected = sorted(set(normalised[mapped.isna()]))[:5]
        raise ValueError(
            f"Could not read '{TARGET}' as a binary label. Unrecognised values: {unexpected}"
        )
    return mapped.astype(int)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select the 17 modelling features and the coerced target."""
    missing = [column for column in FEATURES + [TARGET] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    return df[FEATURES].copy(), coerce_target(df[TARGET])


def make_split(df: pd.DataFrame):
    """Stratified 80/20 split, so the positive rate is preserved on both sides."""
    X, y = split_features_target(df)
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def export_test_data(X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Write the held-out split to test_data.csv, target column included.

    The app needs the labels to compute metrics, so the target must ship with
    the file rather than being stripped out.
    """
    test_df = X_test.copy()
    test_df[TARGET] = y_test.astype(bool).to_numpy()
    test_df.to_csv(TEST_DATA_FILE, index=False)
    return test_df


def load_test_data() -> pd.DataFrame:
    """Read test_data.csv, used as the app's built-in sample."""
    return pd.read_csv(TEST_DATA_FILE)
