from __future__ import annotations
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from IPython.display import Markdown


# ------------------------------
# Main visualization dispatcher
# ------------------------------
def visualize(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    save_plots: bool = False,
    folder: str = "eda_plots",
    max_unique: int = 20,
    plot_type: str = "auto",
    max_cols: int = 2
) -> list[tuple[plt.Figure, Markdown]]:
    """
    Automatically generate exploratory data analysis (EDA) plots with explanations.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if cols is None:
        cols = df.columns.tolist()
    else:
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    if save_plots:
        os.makedirs(folder, exist_ok=True)

    results: list[tuple[plt.Figure, Markdown]] = []

    # Separate numeric and categorical columns
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]

    # --- Numeric columns ---
    for col in numeric_cols:
        try:
            if plot_type in ["auto", "hist"]:
                fig, md = plot_histogram(df, col)
            elif plot_type in ["auto", "box"]:
                fig, md = plot_boxplot(df, col)
            else:
                fig, md = plot_histogram(df, col)

            results.append((fig, md))
            if save_plots:
                fig.savefig(os.path.join(folder, f"{col}_plot.png"))

        except Exception as e:
            results.append((plt.figure(), Markdown(f"❌ Could not plot `{col}`: {e}\n")))

    # --- Categorical columns ---
    for col in categorical_cols:
        try:
            if df[col].nunique() <= max_unique:
                fig, md = plot_countplot(df, col)
            else:
                fig, md = plt.figure(), Markdown(
                    f"⚠️ Skipping `{col}` — too many unique values ({df[col].nunique()}).\n"
                )

            results.append((fig, md))
            if save_plots:
                fig.savefig(os.path.join(folder, f"{col}_plot.png"))

        except Exception as e:
            results.append((plt.figure(), Markdown(f"❌ Could not plot `{col}`: {e}\n")))

    return results


# ------------------------------
# Individual plotting functions
# ------------------------------
def plot_histogram(df: pd.DataFrame, col: str, ax=None) -> tuple[plt.Figure, Markdown]:
    """Plot a histogram for a numeric column."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column '{col}' must be numeric for histogram.")

    fig, ax = (plt.subplots(figsize=(6, 4)) if ax is None else (ax.figure, ax))
    sns.histplot(df[col].dropna(), kde=True, bins=30, color="skyblue", ax=ax)
    ax.set_title(f"Histogram - {col}")

    md = Markdown(
        f"## 📈 Histogram: **{col}**\n"
        f"- Shows distribution of `{col}` values. Useful for detecting skewness, spread, and modality.\n"
    )
    return fig, md


def plot_boxplot(df: pd.DataFrame, col: str, ax=None) -> tuple[plt.Figure, Markdown]:
    """Plot a boxplot for a numeric column."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column '{col}' must be numeric for boxplot.")

    fig, ax = (plt.subplots(figsize=(6, 4)) if ax is None else (ax.figure, ax))
    sns.boxplot(x=df[col], color="lightgreen", ax=ax)
    ax.set_title(f"Boxplot - {col}")

    md = Markdown(
        f"## 📦 Boxplot: **{col}**\n"
        f"- Shows spread and outliers in `{col}`. Good for identifying extreme values.\n"
    )
    return fig, md


def plot_countplot(df: pd.DataFrame, col: str, ax=None) -> tuple[plt.Figure, Markdown]:
    """Plot a countplot for a categorical column."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    fig, ax = (plt.subplots(figsize=(6, 4)) if ax is None else (ax.figure, ax))
    sns.countplot(x=df[col], palette="Set2", order=df[col].value_counts().index, ax=ax)
    ax.set_title(f"Count Plot - {col}")
    ax.tick_params(axis="x", rotation=45)

    md = Markdown(
        f"## 🟦 Count Plot: **{col}**\n"
        f"- Shows frequency of categories in `{col}`. Helpful for understanding class balance.\n"
    )
    return fig, md


def plot_missing_values(df: pd.DataFrame) -> tuple[plt.Figure, Markdown]:
    """Plot missing values per column."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if missing_counts.empty:
        return plt.figure(), Markdown("✅ No missing values detected in dataset.")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=missing_counts.values, y=missing_counts.index, palette="viridis", ax=ax)
    ax.set_title("Missing Values per Column")
    ax.set_xlabel("Number of Missing Values")
    ax.set_ylabel("Column")

    md = Markdown(
        "## 🚨 Missing Values Visualization\n"
        "- Bar chart showing count of missing values per column.\n"
        "- Helps identify problematic features requiring imputation or removal.\n"
    )
    return fig, md


def plot_correlation_matrix(df: pd.DataFrame) -> tuple[plt.Figure, Markdown]:
    """Plot correlation heatmap and provide insights."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    corr = df.corr(numeric_only=True)
    if corr.empty:
        return plt.figure(), Markdown("No numeric columns available for correlation matrix.")

    fig, ax = plt.subplots(figsize=(30, 20))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")

    corr_unstacked = corr.where(~np.eye(corr.shape[0], dtype=bool))
    corr_pairs = corr_unstacked.unstack().dropna()

    if corr_pairs.empty:
        return fig, Markdown("⚠️ Not enough numeric columns for correlation analysis.")

    max_pos_pair, max_pos_value = corr_pairs.idxmax(), corr_pairs.max()
    max_neg_pair, max_neg_value = corr_pairs.idxmin(), corr_pairs.min()

    multi_pairs = corr_pairs[(corr_pairs.abs() > 0.8)]
    multicollinearity_info = (
        ", ".join([f"{a} & {b} ({val:.2f})" for (a, b), val in multi_pairs.items()])
        if not multi_pairs.empty else "None detected"
    )

    md_text = f"""
## 🔗 Correlation Matrix
- Heatmap showing pairwise correlation between numeric features.
- Useful for detecting multicollinearity (features that are too similar).

### 📊 Insights
- **Highest positive correlation:** {max_pos_pair[0]} & {max_pos_pair[1]} → {max_pos_value:.2f}
- **Highest negative correlation:** {max_neg_pair[0]} & {max_neg_pair[1]} → {max_neg_value:.2f}
- **Multicollinearity (|corr| > 0.8):** {multicollinearity_info}
"""
    return fig, Markdown(md_text)


def plot_scatterplot(df: pd.DataFrame, x: str, y: str, hue: str = None) -> tuple[plt.Figure, Markdown]:
    """Plot a scatterplot between two numeric variables."""
    if x not in df.columns or y not in df.columns:
        raise ValueError("Both x and y must be valid column names in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[x]) or not pd.api.types.is_numeric_dtype(df[y]):
        raise TypeError("Both x and y columns must be numeric for scatterplot.")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="viridis", ax=ax)
    ax.set_title(f"Scatter Plot: {x} vs {y}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    md = [
        f"## 📊 Scatter Plot: {x} vs {y}",
        "- Shows relationship between two numeric variables.",
        "- Points may reveal **correlation, clusters, or outliers**."
    ]
    if hue:
        md.append(f"- Colored by **{hue}** for category differentiation.")
    return fig, Markdown("\n".join(md))


def plot_pairplot(df: pd.DataFrame, columns: list = None, hue: str = None) -> tuple[plt.Figure, Markdown]:
    """Plot pairwise relationships across multiple numeric variables."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if columns:
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")

    try:
        g = sns.pairplot(
            df[columns] if columns else df,
            hue=hue,
            diag_kind="kde",
            palette="viridis"
        )
        g.fig.suptitle("Pairplot", y=1.02)

        md = [
            "## 🔍 Pairplot",
            "- Shows pairwise relationships across multiple numeric variables.",
            "- Diagonal shows **distributions**; off-diagonal shows **scatterplots**.",
            "- Useful for detecting **correlations, clusters, and outliers**."
        ]
        if hue:
            md.append(f"- Data points are colored by **{hue}**.")
        return g.fig, Markdown("\n".join(md))

    except Exception as e:
        return plt.figure(), Markdown(f"❌ Could not generate pairplot: {e}")
