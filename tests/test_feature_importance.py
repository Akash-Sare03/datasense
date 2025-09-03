import pytest
import pandas as pd
from datasense.feature_importance import feature_importance_calculate


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
    importance_df, report, task_type = feature_importance_calculate(regression_df, target_col="target")
    assert isinstance(importance_df, pd.DataFrame)
    assert "Feature" in importance_df.columns
    assert "Importance" in importance_df.columns
    assert "Feature Importance Report" in report
    assert task_type == "regression"


def test_feature_importance_classification(classification_df):
    importance_df, report, task_type = feature_importance_calculate(classification_df, target_col="target")
    assert isinstance(importance_df, pd.DataFrame)
    assert "Feature" in importance_df.columns
    assert "Importance" in importance_df.columns
    assert "Feature Importance Report" in report
    assert task_type == "classification"


def test_invalid_target_column(regression_df):
    with pytest.raises(ValueError):
        feature_importance_calculate(regression_df, target_col="not_a_column")


def test_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        feature_importance_calculate(df, target_col="target")
