# test_stats.py

import pytest
import pandas as pd
import numpy as np
from datasense.stats import calculate_statistics
from IPython.display import HTML


@pytest.fixture
def sample_df():
    """Fixture: small numeric + categorical dataset."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 100],   # intentional outlier
        "salary": [3000, 3200, 3500, 4000, 10000],
        "city": ["A", "B", "A", "C", "B"]
    })


def test_full_dataframe(sample_df):
    """Test statistics on all numeric columns."""
    result = calculate_statistics(sample_df)
    assert isinstance(result, HTML)
    assert "Detailed Statistics Report" in result.data


def test_single_column(sample_df):
    """Test statistics on a single column."""
    result = calculate_statistics(sample_df, columns="age")
    assert "Column: age" in result.data


def test_multiple_columns(sample_df):
    """Test statistics on multiple columns."""
    result = calculate_statistics(sample_df, columns=["age", "salary"])
    assert "Column: age" in result.data
    assert "Column: salary" in result.data


def test_invalid_column(sample_df):
    """Test handling of invalid column name."""
    result = calculate_statistics(sample_df, columns="invalid_col")
    assert "Columns not found" in result.data


def test_non_numeric_columns(sample_df):
    """Test behavior with only non-numeric column."""
    result = calculate_statistics(sample_df[["city"]])
    assert "No numeric columns found" in result.data


def test_empty_dataframe():
    """Test behavior with empty dataset."""
    df = pd.DataFrame()
    result = calculate_statistics(df)
    assert "dataset is empty" in result.data


def test_from_csv(tmp_path):
    """Test reading from CSV path."""
    file_path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}).to_csv(file_path, index=False)

    result = calculate_statistics(str(file_path))
    assert "Detailed Statistics Report" in result.data


def test_invalid_input_type():
    """Test unsupported input type."""
    result = calculate_statistics(12345)  # invalid input
    assert "Unsupported input type" in result.data
