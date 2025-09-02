import pandas as pd
import pytest
from datasense.analyze_data import analyze


@pytest.fixture
def sample_df():
    """Provide a small sample DataFrame for testing."""
    return pd.DataFrame({
        "num1": [1, 2, 3, 100, 5],
        "num2": [5, 6, 7, 8, 9],
        "cat": ["a", "b", "a", "c", "b"]
    })


def test_analyze_full_dataframe(sample_df):
    """Test analyze() on full DataFrame (no target_col)."""
    try:
        analyze(sample_df)
    except Exception as e:
        pytest.fail(f"analyze() raised an exception on full DataFrame: {e}")


def test_analyze_single_column(sample_df):
    """Test analyze() focusing on a single column."""
    try:
        analyze(sample_df, target_col="num1")
    except Exception as e:
        pytest.fail(f"analyze() raised an exception on single column: {e}")


def test_analyze_multiple_columns(sample_df):
    """Test analyze() focusing on multiple columns."""
    try:
        analyze(sample_df, target_col=["num1", "num2"])
    except Exception as e:
        pytest.fail(f"analyze() raised an exception on multiple columns: {e}")


def test_analyze_invalid_column(sample_df):
    """Test analyze() with an invalid target column (should not crash)."""
    try:
        analyze(sample_df, target_col="nonexistent")
    except Exception as e:
        pytest.fail(f"analyze() raised an exception on invalid column: {e}")
