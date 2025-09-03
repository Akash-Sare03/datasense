# test_outliers.py

import pytest
import pandas as pd
import numpy as np
from IPython.display import Markdown
from datasense.outliers import detect_outliers, remove_outliers, _to_dataframe


# ---------- Fixtures ----------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "A": [1, 2, 3, 100],   # outlier at 100
        "B": [10, 12, 14, 16], # no strong outlier
        "C": ["x", "y", "z", "w"]  # non-numeric
    })


@pytest.fixture
def empty_df():
    return pd.DataFrame()


# ---------- Tests for _to_dataframe ----------
def test_to_dataframe_from_dataframe(sample_df):
    result = _to_dataframe(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(sample_df)


def test_to_dataframe_from_list():
    data = [{"A": 1, "B": 2}, {"A": 3, "B": 4}]
    df = _to_dataframe(data)
    assert isinstance(df, pd.DataFrame)
    assert "A" in df.columns


def test_to_dataframe_invalid_type():
    with pytest.raises(TypeError):
        _to_dataframe(12345)


# ---------- Tests for detect_outliers ----------
def test_detect_outliers_zscore(sample_df):
    report, figs = detect_outliers(sample_df, method="zscore", threshold=2, visualize=True)
    assert isinstance(report, Markdown)
    assert "Z-Score" in report.data
    assert isinstance(figs, dict)
    assert all(hasattr(fig, "savefig") for fig in figs.values())


def test_detect_outliers_iqr(sample_df):
    report, _ = detect_outliers(sample_df, method="iqr", visualize=False)
    assert "IQR Method Explanation" in report.data
    assert "Summary Table" in report.data


def test_detect_outliers_empty(empty_df):
    report = detect_outliers(empty_df)
    assert "empty" in report.data.lower()


def test_detect_outliers_invalid_method(sample_df):
    with pytest.raises(ValueError):
        detect_outliers(sample_df, method="abc")


# ---------- Tests for remove_outliers ----------
def test_remove_outliers_iqr_remove(sample_df):
    cleaned, md = remove_outliers(sample_df, method="iqr", strategy="remove")
    assert isinstance(cleaned, pd.DataFrame)
    assert isinstance(md, Markdown)
    assert len(cleaned) < len(sample_df)


def test_remove_outliers_iqr_cap(sample_df):
    cleaned, md = remove_outliers(sample_df, method="iqr", strategy="cap")
    assert cleaned["A"].max() <= sample_df["A"].max()
    assert "Capped" in md.data


def test_remove_outliers_iqr_nan(sample_df):
    cleaned, md = remove_outliers(sample_df, method="iqr", strategy="nan")
    assert cleaned.isna().sum().sum() > 0
    assert "NaN" in md.data


def test_remove_outliers_zscore_remove(sample_df):
    cleaned, md = remove_outliers(sample_df, method="zscore", strategy="remove", threshold=2)
    assert len(cleaned) < len(sample_df)
    assert "Z-score" in md.data


def test_remove_outliers_invalid_method(sample_df):
    with pytest.raises(ValueError):
        remove_outliers(sample_df, method="abc")


def test_remove_outliers_invalid_strategy(sample_df):
    with pytest.raises(ValueError):
        remove_outliers(sample_df, method="iqr", strategy="unknown")


def test_remove_outliers_empty(empty_df):
    cleaned, md = remove_outliers(empty_df)
    assert cleaned.empty
    assert "empty" in md.data.lower()
