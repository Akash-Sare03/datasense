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
    x: str | None = None,
    y: str | None = None,
    hue: str | None = None,
    save_plots: bool = False,
    folder: str = "eda_plots",
    max_unique: int = 20,
    plot_type: str = "auto"
) -> list[tuple[plt.Figure, Markdown]]:
    """
    Automatically generate exploratory data analysis (EDA) plots with explanations.

    - If x, y, hue provided → comparative analysis plots.
    - If only cols provided → univariate plots for each column.
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

    # --- Comparative analysis (x, y, hue) ---
    if x and y:
        try:
            fig, md = plot_scatterplot(df, x=x, y=y, hue=hue)
            results.append((fig, md))
            if save_plots:
                fig.savefig(os.path.join(folder, f"scatter_{x}_vs_{y}.png"))
        except Exception as e:
            results.append((plt.figure(), Markdown(f"❌ Could not plot scatterplot: {e}")))

    # --- Univariate plots (hist, box, count) ---
    for col in cols:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                if plot_type in ["auto", "hist"]:
                    fig, md = plot_histogram(df, col)
                elif plot_type in ["box"]:
                    fig, md = plot_boxplot(df, col)
                else:
                    fig, md = plot_histogram(df, col)
            else:
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
def plot_histogram(df: pd.DataFrame, cols: list | str, ax=None) -> tuple[plt.Figure, Markdown]:
    """Plot histogram(s) for one or multiple numeric columns."""
    if isinstance(cols, str):
        cols = [cols]

    fig, ax = plt.subplots(figsize=(6 * len(cols), 4))
    if len(cols) == 1:
        ax = [ax]

    for i, col in enumerate(cols):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric for histogram.")
        sns.histplot(df[col].dropna(), kde=True, bins=30, color="skyblue", ax=ax[i])
        ax[i].set_title(f"Histogram - {col}")

    md = Markdown(
        f"## 📈 Histogram(s)\n- Distribution plots for {', '.join(cols)}.\n"
        "Useful for detecting skewness, spread, and modality.\n"
    )
    return fig, md


def plot_boxplot(df: pd.DataFrame, cols: list | str, ax=None) -> tuple[plt.Figure, Markdown]:
    """Plot boxplot(s) for one or multiple numeric columns."""
    if isinstance(cols, str):
        cols = [cols]

    fig, ax = plt.subplots(figsize=(6 * len(cols), 4))
    if len(cols) == 1:
        ax = [ax]

    for i, col in enumerate(cols):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric for boxplot.")
        sns.boxplot(x=df[col], color="lightgreen", ax=ax[i])
        ax[i].set_title(f"Boxplot - {col}")

    md = Markdown(
        f"## 📦 Boxplot(s)\n- Spread and outliers for {', '.join(cols)}.\n"
        "Good for identifying extreme values.\n"
    )
    return fig, md


def plot_countplot(df: pd.DataFrame, cols: list | str, ax=None) -> tuple[plt.Figure, Markdown]:
    """Plot countplot(s) for one or multiple categorical columns."""
    if isinstance(cols, str):
        cols = [cols]

    fig, ax = plt.subplots(figsize=(6 * len(cols), 4))
    if len(cols) == 1:
        ax = [ax]

    for i, col in enumerate(cols):
        sns.countplot(x=df[col], palette="Set2", order=df[col].value_counts().index, ax=ax[i])
        ax[i].set_title(f"Count Plot - {col}")
        ax[i].tick_params(axis="x", rotation=45)

    md = Markdown(
        f"## 🟦 Count Plot(s)\n- Frequency distribution for {', '.join(cols)}.\n"
    )
    return fig, md


def plot_missing_values(df: pd.DataFrame) -> tuple[plt.Figure, Markdown]:
    """Plot missing values per column."""
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if missing_counts.empty:
        return plt.figure(), Markdown("✅ No missing values detected in dataset.")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=missing_counts.values, y=missing_counts.index, palette="viridis", ax=ax)
    ax.set_title("Missing Values per Column")

    md = Markdown(
        "## 🚨 Missing Values Visualization\n"
        "- Count of missing values per column.\n"
    )
    return fig, md


def plot_correlation_matrix(df: pd.DataFrame) -> tuple[plt.Figure, Markdown]:
    """Plot correlation heatmap with insights."""
    corr = df.corr(numeric_only=True)
    if corr.empty:
        return plt.figure(), Markdown("No numeric columns available for correlation matrix.")

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Matrix")

    md = Markdown("## 🔗 Correlation Matrix\n- Pairwise correlation between numeric features.\n")
    return fig, md


def plot_scatterplot(df: pd.DataFrame, x: str, y: str, hue: str = None) -> tuple[plt.Figure, Markdown]:
    """Plot a scatterplot between two numeric variables, with optional hue."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="viridis", ax=ax)
    ax.set_title(f"Scatter Plot: {x} vs {y}")

    md = Markdown(
        f"## 📊 Scatter Plot: {x} vs {y}\n- Relationship between two numeric variables.\n"
        + (f"- Colored by **{hue}**.\n" if hue else "")
    )
    return fig, md


def plot_pairplot(df: pd.DataFrame, columns: list = None, hue: str = None) -> tuple[plt.Figure, Markdown]:
    """Plot pairwise relationships across multiple numeric variables."""
    g = sns.pairplot(
        df[columns] if columns else df,
        hue=hue,
        diag_kind="kde",
        palette="viridis"
    )
    g.fig.suptitle("Pairplot", y=1.02)

    md = Markdown("## 🔍 Pairplot\n- Pairwise relationships across multiple variables.\n")
    return g.fig, md
