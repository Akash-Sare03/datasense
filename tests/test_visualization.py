# test_visualization.py

import pytest
import pandas as pd
import numpy as np
import matplotlib
# Set non-interactive backend to avoid TclError
matplotlib.use('Agg')  # Use Agg backend for testing
import matplotlib.pyplot as plt
from datasense.visualization import (
    visualize,
    plot_histogram,
    plot_boxplot,
    plot_countplot,
    plot_violinplot,
    plot_stacked_barplot,
    plot_missing_values,
    plot_correlation_matrix,
    plot_scatterplot,
    plot_pairplot,
    plot_facet_grid,
)


@pytest.fixture
def sample_df():
    """Fixture: Small mixed DataFrame with numeric + categorical."""
    return pd.DataFrame({
        "age": [23, 45, 31, 35, 62, 28],
        "salary": [40000, 50000, 42000, 60000, 58000, 45000],
        "score": [85, 92, 78, 88, 95, 82],
        "dept": ["HR", "IT", "Finance", "IT", "Finance", "HR"],
        "experience": [2, 8, 5, 7, 15, 3]
    })


@pytest.fixture
def sample_df_with_missing():
    """Fixture: DataFrame with missing values."""
    return pd.DataFrame({
        "age": [23, 45, np.nan, 35, 62, np.nan],
        "salary": [40000, 50000, 42000, 60000, 58000, 45000],
        "dept": ["HR", "IT", "Finance", "IT", "Finance", "HR"],
    })


@pytest.fixture
def sample_large_df():
    """Fixture: Larger DataFrame for more comprehensive testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feature1": np.random.normal(0, 1, n),
        "feature2": np.random.normal(5, 2, n),
        "feature3": np.random.exponential(1, n),
        "category": np.random.choice(["A", "B", "C"], n),
        "group": np.random.choice(["X", "Y"], n)
    })


@pytest.fixture
def sample_numeric_df():
    """Fixture: DataFrame with only numeric columns for pairplot testing."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "var1": np.random.normal(0, 1, n),
        "var2": np.random.normal(5, 2, n),
        "var3": np.random.exponential(1, n),
        "var4": np.random.uniform(0, 10, n),
        "cluster": np.random.randint(1, 4, n)  # Numeric cluster for hue
    })


# --------------------------
# Dispatcher tests
# --------------------------
def test_visualize_numeric_and_categorical(sample_df):
    results = visualize(sample_df)
    assert isinstance(results, list)
    for fig, md in results:
        assert isinstance(fig, plt.Figure)
        assert hasattr(md, 'data')  # Check it's a Markdown object


def test_visualize_with_missing_column(sample_df):
    with pytest.raises(ValueError):
        visualize(sample_df, cols=["nonexistent"])


def test_visualize_with_non_dataframe():
    with pytest.raises(TypeError):
        visualize([1, 2, 3])  # Not a DataFrame


def test_visualize_compare_mode(sample_df):
    # Test comparative analysis mode
    results = visualize(sample_df, x='age', y='salary', hue='dept', compare=True)
    assert isinstance(results, list)
    for fig, md in results:
        assert isinstance(fig, plt.Figure)
        assert hasattr(md, 'data')


def test_visualize_compare_mode_single_column(sample_df):
    # Test comparative analysis with single column
    results = visualize(sample_df, x='dept', compare=True)
    assert isinstance(results, list)
    for fig, md in results:
        assert isinstance(fig, plt.Figure)
        assert hasattr(md, 'data')


def test_visualize_compare_mode_pairplot(sample_numeric_df):
    # Test comparative analysis with multiple numeric columns for pairplot
    results = visualize(sample_numeric_df, cols=['var1', 'var2', 'var3'], compare=True)
    assert isinstance(results, list)
    for fig, md in results:
        assert isinstance(fig, plt.Figure)
        assert hasattr(md, 'data')


# --------------------------
# Individual plot tests
# --------------------------
def test_plot_histogram_valid(sample_df):
    fig, md = plot_histogram(sample_df, "age")
    assert isinstance(fig, plt.Figure)
    assert "Histogram" in md.data


def test_plot_histogram_with_hue(sample_df):
    fig, md = plot_histogram(sample_df, "age", hue="dept")
    assert isinstance(fig, plt.Figure)
    assert "Histogram" in md.data
    assert "dept" in md.data


def test_plot_histogram_invalid_col(sample_df):
    with pytest.raises(ValueError):
        plot_histogram(sample_df, "nonexistent")

    with pytest.raises(TypeError):
        plot_histogram(sample_df, "dept")  # categorical column


def test_plot_boxplot_valid(sample_df):
    fig, md = plot_boxplot(sample_df, "salary")
    assert isinstance(fig, plt.Figure)
    assert "Boxplot" in md.data


def test_plot_boxplot_with_hue(sample_df):
    fig, md = plot_boxplot(sample_df, "salary", hue="dept")
    assert isinstance(fig, plt.Figure)
    assert "Boxplot" in md.data
    assert "dept" in md.data


def test_plot_countplot_valid(sample_df):
    fig, md = plot_countplot(sample_df, "dept")
    assert isinstance(fig, plt.Figure)
    assert "Count Plot" in md.data


def test_plot_countplot_with_hue(sample_df):
    fig, md = plot_countplot(sample_df, "dept", hue="experience")
    assert isinstance(fig, plt.Figure)
    assert "Count Plot" in md.data


def test_plot_countplot_invalid_col(sample_df):
    with pytest.raises(ValueError):
        plot_countplot(sample_df, "nonexistent")


def test_plot_violinplot_valid(sample_df):
    fig, md = plot_violinplot(sample_df, "dept", "salary")
    assert isinstance(fig, plt.Figure)
    assert "Violin Plot" in md.data


def test_plot_violinplot_invalid_cols(sample_df):
    with pytest.raises(ValueError):
        plot_violinplot(sample_df, "nonexistent", "salary")
    with pytest.raises(TypeError):
        plot_violinplot(sample_df, "dept", "dept")  # y must be numeric


def test_plot_stacked_barplot_valid(sample_df):
    fig, md = plot_stacked_barplot(sample_df, "dept", "experience")
    assert isinstance(fig, plt.Figure)
    assert "Stacked Bar Plot" in md.data


def test_plot_stacked_barplot_invalid_cols(sample_df):
    with pytest.raises(ValueError):
        plot_stacked_barplot(sample_df, "nonexistent", "dept")


def test_plot_missing_values(sample_df_with_missing):
    fig, md = plot_missing_values(sample_df_with_missing)
    assert isinstance(fig, plt.Figure)
    assert isinstance(md.data, str)


def test_plot_missing_values_no_missing(sample_df):
    fig, md = plot_missing_values(sample_df)
    assert "No missing values" in md.data or "✅" in md.data


def test_plot_correlation_matrix(sample_df):
    fig, md = plot_correlation_matrix(sample_df)
    assert isinstance(fig, plt.Figure)
    assert "Correlation Matrix" in md.data


def test_plot_correlation_matrix_different_methods(sample_df):
    # Test different correlation methods
    fig, md = plot_correlation_matrix(sample_df, method='spearman')
    assert isinstance(fig, plt.Figure)
    assert "spearman" in md.data.lower()

    fig, md = plot_correlation_matrix(sample_df, method='kendall')
    assert isinstance(fig, plt.Figure)
    assert "kendall" in md.data.lower()


def test_plot_correlation_matrix_no_numeric():
    df = pd.DataFrame({"dept": ["HR", "IT", "Finance"]})
    fig, md = plot_correlation_matrix(df)
    assert "no numeric columns" in md.data.lower()


def test_plot_scatterplot_valid(sample_df):
    fig, md = plot_scatterplot(sample_df, "age", "salary")
    assert isinstance(fig, plt.Figure)
    assert "Scatter Plot" in md.data


def test_plot_scatterplot_with_hue(sample_df):
    fig, md = plot_scatterplot(sample_df, "age", "salary", hue="dept")
    assert isinstance(fig, plt.Figure)
    assert "Scatter Plot" in md.data
    assert "dept" in md.data


def test_plot_scatterplot_with_size_and_style(sample_df):
    fig, md = plot_scatterplot(sample_df, "age", "salary", size="experience", style="dept")
    assert isinstance(fig, plt.Figure)
    assert "Scatter Plot" in md.data


def test_plot_scatterplot_invalid_cols(sample_df):
    with pytest.raises(ValueError):
        plot_scatterplot(sample_df, "x", "salary")
    with pytest.raises(TypeError):
        plot_scatterplot(sample_df, "dept", "salary")


def test_plot_pairplot_valid(sample_numeric_df):
    fig, md = plot_pairplot(sample_numeric_df, columns=["var1", "var2"])
    assert isinstance(fig, plt.Figure)
    assert "Pairplot" in md.data


def test_plot_pairplot_with_numeric_hue(sample_numeric_df):
    # Use numeric column for hue in pairplot
    fig, md = plot_pairplot(sample_numeric_df, columns=["var1", "var2"], hue="cluster")
    assert isinstance(fig, plt.Figure)
    assert "Pairplot" in md.data


def test_plot_pairplot_corner_mode(sample_numeric_df):
    fig, md = plot_pairplot(sample_numeric_df, columns=["var1", "var2", "var3"], corner=True)
    assert isinstance(fig, plt.Figure)
    assert "Pairplot" in md.data
    assert "lower triangle" in md.data.lower()


def test_plot_pairplot_auto_columns(sample_numeric_df):
    # Test without specifying columns (should auto-select numeric)
    fig, md = plot_pairplot(sample_numeric_df)
    assert isinstance(fig, plt.Figure)
    assert "Pairplot" in md.data


def test_plot_pairplot_invalid_col(sample_df):
    with pytest.raises(ValueError):
        plot_pairplot(sample_df, columns=["invalid_col"])


def test_plot_facet_grid_valid(sample_df):
    fig, md = plot_facet_grid(sample_df, "age", "salary", col="dept")
    assert isinstance(fig, plt.Figure)
    assert "Facet Grid" in md.data


def test_plot_facet_grid_with_row_and_hue(sample_df):
    # Create a DataFrame with more groups for row testing
    df = sample_df.copy()
    df['seniority'] = np.where(df['age'] > 40, 'Senior', 'Junior')
    
    fig, md = plot_facet_grid(df, "age", "salary", col="dept", row="seniority", hue="dept")
    assert isinstance(fig, plt.Figure)
    assert "Facet Grid" in md.data


def test_plot_facet_grid_different_kinds(sample_large_df):
    # Test different plot kinds
    fig, md = plot_facet_grid(sample_large_df, "feature1", "feature2", col="category", kind="scatter")
    assert isinstance(fig, plt.Figure)
    
    fig, md = plot_facet_grid(sample_large_df, "feature1", "feature2", col="category", kind="line")
    assert isinstance(fig, plt.Figure)


def test_plot_facet_grid_invalid_cols(sample_df):
    # Test that invalid columns raise ValueError
    with pytest.raises(ValueError, match="not found in DataFrame"):
        plot_facet_grid(sample_df, "nonexistent", "salary", col="dept")
    
    with pytest.raises(ValueError, match="not found in DataFrame"):
        plot_facet_grid(sample_df, "age", "salary", col="nonexistent")


def test_plot_facet_grid_invalid_kind(sample_df):
    with pytest.raises(ValueError, match="must be"):
        plot_facet_grid(sample_df, "age", "salary", col="dept", kind="invalid_kind")


# --------------------------
# Edge case tests
# --------------------------
def test_visualize_empty_dataframe():
    df = pd.DataFrame()
    results = visualize(df)
    # Should handle empty DataFrame gracefully
    assert isinstance(results, list)


def test_visualize_single_column_string():
    df = pd.DataFrame({"col": ["a", "b", "c"]})
    results = visualize(df)
    assert isinstance(results, list)


def test_visualize_all_numeric(sample_large_df):
    # Test with only numeric columns
    numeric_df = sample_large_df.select_dtypes(include=[np.number])
    results = visualize(numeric_df)
    assert isinstance(results, list)


def test_visualize_all_categorical():
    # Test with only categorical columns
    df = pd.DataFrame({
        "cat1": ["A", "B", "A", "B"],
        "cat2": ["X", "Y", "X", "Y"]
    })
    results = visualize(df)
    assert isinstance(results, list)


# --------------------------
# Test error handling
# --------------------------
def test_plot_functions_with_nonexistent_hue(sample_df):
    # Test that functions handle non-existent hue gracefully
    # This should not raise an exception but might show an error message
    fig, md = plot_histogram(sample_df, "age", hue="nonexistent")
    assert isinstance(fig, plt.Figure)
    # The function should handle this gracefully, either by working or showing an error message


def test_plot_functions_with_too_many_categories():
    # Test with column that has too many unique values for hue
    df = pd.DataFrame({
        "value": range(50),
        "category": [f"cat_{i}" for i in range(50)]  # 50 unique categories
    })
    
    fig, md = plot_histogram(df, "value", hue="category")
    assert isinstance(fig, plt.Figure)


def test_plot_pairplot_with_categorical_hue_graceful(sample_df):
    # Test that pairplot handles categorical hue gracefully (should not crash)
    # This might return an error message but should not raise an exception
    fig, md = plot_pairplot(sample_df, columns=["age", "salary"], hue="dept")
    assert isinstance(fig, plt.Figure)
    # The function should handle categorical hue gracefully


# --------------------------
# Test specific error messages
# --------------------------
def test_plot_pairplot_no_numeric_columns():
    # Test with DataFrame that has no numeric columns
    df = pd.DataFrame({
        "cat1": ["A", "B", "C"],
        "cat2": ["X", "Y", "Z"]
    })
    fig, md = plot_pairplot(df)
    assert "no numeric columns" in md.data.lower()


def test_plot_correlation_matrix_custom_figsize(sample_df):
    # Test with custom figure size
    fig, md = plot_correlation_matrix(sample_df, figsize=(8, 6))
    assert isinstance(fig, plt.Figure)
    assert "Correlation Matrix" in md.data