# test_timeseries.py

import pytest
import pandas as pd
import numpy as np
from datasense.timeseries import analyze_timeseries


@pytest.fixture
def sample_ts_df():
    """Fixture: small time-series dataset with daily frequency."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    values = [10, 12, 13, 15, 16, 18, 19, 20, 21, 22]
    return pd.DataFrame({"date": dates, "sales": values})


def test_valid_timeseries(sample_ts_df):
    """Test normal execution with valid time-series."""
    # Just check it runs without errors (since function outputs via display).
    analyze_timeseries(sample_ts_df, date_col="date", target_col="sales")


def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    df = pd.DataFrame()
    analyze_timeseries(df, date_col="date", target_col="sales")  # Should not crash


def test_missing_columns(sample_ts_df):
    """Test missing date_col or target_col."""
    analyze_timeseries(sample_ts_df, date_col="missing", target_col="sales")
    analyze_timeseries(sample_ts_df, date_col="date", target_col="missing")


def test_invalid_date_column(sample_ts_df):
    """Test when date column cannot be parsed as datetime."""
    df = sample_ts_df.copy()
    df["date"] = "not_a_date"
    analyze_timeseries(df, date_col="date", target_col="sales")


def test_non_numeric_target(sample_ts_df):
    """Test non-numeric target column."""
    df = sample_ts_df.copy()
    df["sales"] = ["a"] * len(df)
    analyze_timeseries(df, date_col="date", target_col="sales")


def test_all_invalid_dates(sample_ts_df):
    """Test when all date values are invalid."""
    df = sample_ts_df.copy()
    df["date"] = None  # forces all NaT
    analyze_timeseries(df, date_col="date", target_col="sales")


def test_all_nan_target(sample_ts_df):
    """Test when target column has no valid numeric values."""
    df = sample_ts_df.copy()
    df["sales"] = np.nan
    analyze_timeseries(df, date_col="date", target_col="sales")
