import pandas as pd
import numpy as np
import pytest
from IPython.display import Markdown
from datasense.outliers import _to_dataframe, detect_outliers, remove_outliers


@pytest.fixture
def sample_df():
    """Simple dataset with numeric values and some potential outliers."""
    return pd.DataFrame({
        "A": [1, 2, 3, 100],    # Outlier at 100
        "B": [5, 6, 7, 8],      # No strong outlier
        "C": ["x", "y", "z", "w"]  # Non-numeric column
    })


# ========== _to_dataframe ==========
def test_to_dataframe_with_dataframe(sample_df):
    df = _to_dataframe(sample_df)
    assert isinstance(df, pd.DataFrame)
    assert df.equals(sample_df)


def test_to_dataframe_with_dict():
    data = {"A": [1, 2], "B": [3, 4]}
    df = _to_dataframe(data)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["A", "B"]


def test_to_dataframe_invalid_type():
    with pytest.raises(TypeError):
        _to_dataframe(12345)


# ========== detect_outliers ==========
def test_detect_outliers_zscore(sample_df):
    report, figs = detect_outliers(sample_df, method="zscore", visualize=True)
    assert isinstance(report, Markdown)
    assert "Z-Score Method Explanation" in report.data
    assert isinstance(figs, dict)
    assert "A" in figs  # should produce figure for numeric columns


def test_detect_outliers_iqr_no_visual(sample_df):
    report = detect_outliers(sample_df, method="iqr", visualize=False)
    assert isinstance(report, Markdown)
    assert "IQR Method Explanation" in report.data


def test_detect_outliers_empty():
    df = pd.DataFrame()
    report = detect_outliers(df)
    assert "The dataset is empty" in report.data


def test_detect_outliers_no_numeric():
    df = pd.DataFrame({"A": ["x", "y", "z"]})
    report = detect_outliers(df)
    assert "No numeric columns found" in report.data


def test_detect_outliers_invalid_method(sample_df):
    """Invalid method should return error message in Markdown."""
    # The function returns an error message, not raises an exception
    report = detect_outliers(sample_df, method="invalid", visualize=False)
    assert isinstance(report, Markdown)
    assert "error" in report.data.lower() or "invalid" in report.data.lower()


# ========== remove_outliers ==========
def test_remove_outliers_iqr_remove(sample_df):
    cleaned, report = remove_outliers(sample_df, method="iqr", strategy="remove")
    assert isinstance(cleaned, pd.DataFrame)
    assert "Removed" in report.data or "Capped" in report.data


def test_remove_outliers_iqr_cap(sample_df):
    cleaned, report = remove_outliers(sample_df, method="iqr", strategy="cap")
    assert "Capped" in report.data


def test_remove_outliers_iqr_nan(sample_df):
    cleaned, report = remove_outliers(sample_df, method="iqr", strategy="nan")
    assert cleaned.isnull().sum().sum() > 0
    assert "NaN" in report.data


def test_remove_outliers_zscore_remove(sample_df):
    cleaned, report = remove_outliers(sample_df, method="zscore", strategy="remove", threshold=2)
    assert "Z-score" in report.data


def test_remove_outliers_zscore_cap(sample_df):
    cleaned, report = remove_outliers(sample_df, method="zscore", strategy="cap", threshold=2)
    assert "Capped" in report.data


def test_remove_outliers_zscore_nan(sample_df):
    cleaned, report = remove_outliers(sample_df, method="zscore", strategy="nan", threshold=2)
    assert "NaN" in report.data


def test_remove_outliers_invalid_method(sample_df):
    """Invalid method should return original DataFrame with error message."""
    # The function returns an error message, not raises an exception
    result_df, report = remove_outliers(sample_df, method="invalid")
    
    # The original DataFrame should be returned unchanged
    assert result_df.equals(sample_df)
    
    # The report should contain an error message
    assert "error" in report.data.lower() or "invalid" in report.data.lower()


def test_remove_outliers_invalid_strategy(sample_df):
    """Invalid strategy should return original DataFrame with error message."""
    # The function returns an error message, not raises an exception
    result_df, report = remove_outliers(sample_df, method="iqr", strategy="invalid")
    
    # The original DataFrame should be returned unchanged
    assert result_df.equals(sample_df)
    
    # The report should contain an error message
    assert "error" in report.data.lower() or "invalid" in report.data.lower()