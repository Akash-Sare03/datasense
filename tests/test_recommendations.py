import pytest
import pandas as pd
import numpy as np
from IPython.display import Markdown
from datasense.recommendations import _to_dataframe, generate_recommendations


# --------------------
# Tests for _to_dataframe
# --------------------
def test_to_dataframe_with_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = _to_dataframe(df)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(df)


def test_to_dataframe_with_list():
    data = [{"a": 1}, {"a": 2}]
    df = _to_dataframe(data)
    assert isinstance(df, pd.DataFrame)
    assert "a" in df.columns


def test_to_dataframe_with_dict():
    data = {"a": [1, 2, 3]}
    df = _to_dataframe(data)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 1)


def test_to_dataframe_with_invalid_type():
    with pytest.raises(TypeError):
        _to_dataframe(12345)


# --------------------
# Tests for generate_recommendations
# --------------------
def test_generate_recommendations_valid_df():
    df = pd.DataFrame({
        "num": [1, 2, 3, 100],  # outlier
        "cat": ["a", "b", "a", None],  # missing + categorical
        "const": [1, 1, 1, 1]  # constant column
    })
    md = generate_recommendations(df)
    assert isinstance(md, Markdown)
    text = md.data
    # check that major sections are included
    assert "Missing Values" in text
    assert "Constant Columns" in text
    assert "Categorical Encoding" in text
    assert "Outliers" in text


def test_generate_recommendations_empty_df():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        generate_recommendations(df)


def test_generate_recommendations_with_list():
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    md = generate_recommendations(data)
    assert isinstance(md, Markdown)
    assert "Data Recommendations Report" in md.data


def test_generate_recommendations_with_dict():
    data = {"a": [1, 2], "b": [3, 4]}
    md = generate_recommendations(data)
    assert isinstance(md, Markdown)
    assert "Correlation Check" in md.data


def test_generate_recommendations_skewness_and_corr():
    df = pd.DataFrame({
        "x": np.random.exponential(scale=2, size=1000),  # skewed
        "y": np.arange(1000)  # correlated with itself
    })
    md = generate_recommendations(df)
    text = md.data
    assert "Skewed Features" in text
    assert "Correlation Check" in text
