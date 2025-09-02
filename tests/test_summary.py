import pytest
import pandas as pd
import numpy as np
from IPython.display import Markdown
from datasense.summary import _to_dataframe, summarize_dataset


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


def test_to_dataframe_invalid_type():
    with pytest.raises(TypeError):
        _to_dataframe(42)


# --------------------
# Tests for summarize_dataset
# --------------------
def test_summarize_valid_df():
    df = pd.DataFrame({
        "num": [1, 2, 3, 4, 100],  # numeric column
        "cat": ["x", "y", "z", "y", "x"],  # categorical
        "dup": [1, 1, 1, 1, 1]  # constant column
    })
    md = summarize_dataset(df)
    assert isinstance(md, Markdown)
    text = md.data
    # Core sections should exist
    assert "Shape" in text
    assert "Missing Values" in text
    assert "Duplicate Rows" in text
    assert "Unique Values" in text
    assert "Numeric Summary" in text
    assert "Sample Rows" in text


def test_summarize_empty_df():
    df = pd.DataFrame()
    md = summarize_dataset(df)
    assert isinstance(md, Markdown)
    assert "empty" in md.data.lower()


def test_summarize_with_list():
    data = [{"a": 1, "b": 2}, {"a": 3, "b": None}]
    md = summarize_dataset(data)
    assert isinstance(md, Markdown)
    assert "Missing Values" in md.data


def test_summarize_with_dict():
    data = {"col1": [1, 2, 3], "col2": ["a", "b", "c"]}
    md = summarize_dataset(data)
    assert isinstance(md, Markdown)
    assert "Unique Values" in md.data


def test_summarize_without_sample():
    df = pd.DataFrame({"a": [1, 2, 3]})
    md = summarize_dataset(df, include_sample=False)
    text = md.data
    assert "Sample Rows" not in text


def test_summarize_numeric_stats():
    df = pd.DataFrame({
        "val": np.arange(10),
        "val2": np.random.randn(10)
    })
    md = summarize_dataset(df)
    text = md.data
    assert "Mean" in text
    assert "Median" in text
    assert "Std Dev" in text
    assert "Min" in text
    assert "Max" in text
