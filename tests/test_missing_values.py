import pandas as pd
import pytest
from IPython.display import Markdown
from datasense.missing_values import find_missing_values, handle_missing_values


@pytest.fixture
def sample_df():
    """Small dataset with missing values."""
    return pd.DataFrame({
        "A": [1, 2, None, 4],
        "B": [None, "x", "y", None],
        "C": [10, 20, 30, 40]
    })


def test_find_missing_values_with_missing(sample_df):
    """Should generate a Markdown report when missing values exist."""
    report = find_missing_values(sample_df)
    assert isinstance(report, Markdown)
    assert "Missing Values Report" in report.data
    assert "Total Missing Values" in report.data


def test_find_missing_values_no_missing():
    """Should return a report with 'No missing values' when dataset is complete."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    report = find_missing_values(df)
    assert "No missing values found" in report.data or "✅" in report.data


def test_find_missing_values_empty():
    """Empty dataset should return a warning Markdown."""
    df = pd.DataFrame()
    report = find_missing_values(df)
    assert "The dataset is empty" in report.data or "empty" in report.data.lower()


def test_handle_missing_values_drop(sample_df):
    """Dropping missing rows should reduce dataset size."""
    cleaned_df, report = handle_missing_values(sample_df, method="drop")
    assert len(cleaned_df) < len(sample_df)
    assert "dropped" in report.data.lower()


def test_handle_missing_values_mean(sample_df):
    """Mean imputation should replace NaNs in numeric columns."""
    cleaned_df, report = handle_missing_values(sample_df, method="mean")
    assert cleaned_df["A"].isnull().sum() == 0
    assert "mean" in report.data.lower()


def test_handle_missing_values_median(sample_df):
    """Median imputation should replace NaNs in numeric columns."""
    cleaned_df, report = handle_missing_values(sample_df, method="median")
    assert cleaned_df["A"].isnull().sum() == 0
    assert "median" in report.data.lower()


def test_handle_missing_values_mode(sample_df):
    """Mode imputation should replace NaNs in all columns."""
    cleaned_df, report = handle_missing_values(sample_df, method="mode")
    assert cleaned_df.isnull().sum().sum() == 0
    assert "mode" in report.data.lower()


def test_handle_missing_values_constant(sample_df):
    """Constant imputation should replace NaNs with provided value."""
    cleaned_df, report = handle_missing_values(sample_df, method="constant", value=-1)
    assert (cleaned_df == -1).sum().sum() > 0
    assert "constant" in report.data.lower()


def test_handle_missing_values_invalid_method(sample_df):
    """Invalid method should return error message in Markdown."""
    # This is the corrected test - the function returns an error message, not raises an exception
    result_df, report = handle_missing_values(sample_df, method="invalid")
    
    # The original DataFrame should be returned unchanged
    assert result_df.equals(sample_df)
    
    # The report should contain an error message
    assert "error" in report.data.lower() or "invalid" in report.data.lower()