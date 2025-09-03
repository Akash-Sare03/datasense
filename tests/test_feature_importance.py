import pytest
import pandas as pd
from datasense.feature_importance import feature_importance_calculate
from IPython.display import Markdown


@pytest.fixture
def regression_df():
    return pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6],
        "feature2": [2, 4, 6, 8, 10, 12],
        "target":   [5, 10, 15, 20, 25, 30],  # Numeric target → regression
    })


@pytest.fixture
def classification_df():
    return pd.DataFrame({
        "feature1": [1, 2, 1, 2, 3, 3],
        "feature2": [10, 20, 30, 10, 20, 30],
        "target":   ["A", "A", "B", "B", "C", "C"],  # Categorical target → classification
    })


def test_feature_importance_regression(regression_df):
    report = feature_importance_calculate(regression_df, target_col="target")
    assert isinstance(report, Markdown)
    assert "Feature Importance" in str(report.data)


def test_feature_importance_classification(classification_df):
    report = feature_importance_calculate(classification_df, target_col="target")
    assert isinstance(report, Markdown)
    assert "Feature Importance" in str(report.data)


def test_invalid_target_column(regression_df):
    with pytest.raises(KeyError):
        feature_importance_calculate(regression_df, target_col="not_a_column")


def test_empty_dataframe():
    df = pd.DataFrame()
    report = feature_importance_calculate(df, target_col="target")
    assert isinstance(report, Markdown)
    assert "empty" in str(report.data).lower()
