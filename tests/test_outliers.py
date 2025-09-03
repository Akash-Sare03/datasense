import pytest
import pandas as pd
from datasense.outliers import detect_outliers, remove_outliers
from IPython.display import Markdown

# Sample DataFrame
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "A": [1, 2, 3, 100],
        "B": [10, 12, 14, 16],
        "C": ["x", "y", "z", "w"]
    })


def test_detect_outliers_iqr(sample_df):
    result = detect_outliers(sample_df, method="iqr", visualize=False)
    # Depending on visualize flag, result can be Markdown OR (Markdown, dict)
    assert isinstance(result, Markdown) or (
        isinstance(result, tuple) and isinstance(result[0], Markdown)
    )


def test_detect_outliers_invalid_method(sample_df):
    result = detect_outliers(sample_df, method="invalid", visualize=False)
    assert isinstance(result, Markdown)
    assert "Invalid method" in str(result.data)


def test_remove_outliers_zscore_remove(sample_df):
    cleaned, report = remove_outliers(sample_df, method="zscore", threshold=1.0)
    assert len(cleaned) < len(sample_df)  # Now should remove outliers
    assert isinstance(report, Markdown)


def test_remove_outliers_invalid_method(sample_df):
    _, report = remove_outliers(sample_df, method="invalid")
    assert isinstance(report, Markdown)
    assert "Invalid method" in str(report.data)


def test_remove_outliers_invalid_strategy(sample_df):
    _, report = remove_outliers(sample_df, method="iqr", strategy="bad_strategy")
    assert isinstance(report, Markdown)
    assert "Invalid strategy" in str(report.data)
