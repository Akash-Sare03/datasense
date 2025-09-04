from __future__ import annotations
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from IPython.display import Markdown
from typing import Union, List, Optional, Tuple


# ------------------------------
# Main visualization dispatcher
# ------------------------------
def visualize(
    df: pd.DataFrame,
    cols: Union[List[str], str, None] = None,
    save_plots: bool = False,
    folder: str = "eda_plots",
    max_unique: int = 20,
    plot_type: str = "auto",
    max_cols: int = 2,
    x: Optional[str] = None,
    y: Optional[str] = None,
    hue: Optional[str] = None,
    compare: bool = False
) -> List[Tuple[plt.Figure, Markdown]]:
    """
    Automatically generate exploratory data analysis (EDA) plots with explanations.
    Supports both single column analysis and comparative analysis with x, y, hue parameters.
    
    Args:
        df: Input DataFrame
        cols: Column(s) to analyze. Can be a single column name or list of columns
        save_plots: Whether to save plots to files
        folder: Folder to save plots if save_plots is True
        max_unique: Maximum unique values for categorical columns
        plot_type: Type of plot to generate ("auto", "hist", "box", "scatter", "pair")
        max_cols: Maximum number of columns for subplots
        x: Column name for x-axis in comparative plots
        y: Column name for y-axis in comparative plots
        hue: Column name for hue (color grouping) in comparative plots
        compare: Whether to generate comparative plots instead of individual plots
    
    Returns:
        List of tuples containing (figure, markdown_description)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Handle different input formats for cols parameter
    if cols is None:
        cols = df.columns.tolist()
    elif isinstance(cols, str):
        cols = [cols]
    
    # Validate columns exist
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    if save_plots:
        os.makedirs(folder, exist_ok=True)

    results: List[Tuple[plt.Figure, Markdown]] = []

    # Handle comparative analysis
    if compare:
        try:
            if x and y:
                # Scatter plot with optional hue
                fig, md = plot_scatterplot(df, x=x, y=y, hue=hue)
                results.append((fig, md))
                if save_plots:
                    fig.savefig(os.path.join(folder, f"scatter_{x}_vs_{y}.png"))
            
            elif x and not y:
                # Distribution comparison with hue
                if pd.api.types.is_numeric_dtype(df[x]):
                    if hue and df[hue].nunique() <= max_unique:
                        fig, md = plot_violinplot(df, x=hue, y=x)
                    else:
                        fig, md = plot_histogram(df, x)
                else:
                    if hue and df[hue].nunique() <= max_unique:
                        fig, md = plot_stacked_barplot(df, x=x, hue=hue)
                    else:
                        fig, md = plot_countplot(df, x)
                results.append((fig, md))
                if save_plots:
                    fig.savefig(os.path.join(folder, f"distribution_{x}.png"))
            
            elif not x and y:
                # Distribution of y column with hue grouping
                if pd.api.types.is_numeric_dtype(df[y]):
                    if hue and df[hue].nunique() <= max_unique:
                        fig, md = plot_violinplot(df, x=hue, y=y)
                    else:
                        fig, md = plot_histogram(df, y)
                results.append((fig, md))
                if save_plots:
                    fig.savefig(os.path.join(folder, f"distribution_{y}.png"))
            
            # Pairplot for multiple columns
            if len(cols) > 1 and all(pd.api.types.is_numeric_dtype(df[col]) for col in cols):
                fig, md = plot_pairplot(df, columns=cols, hue=hue)
                results.append((fig, md))
                if save_plots:
                    fig.savefig(os.path.join(folder, f"pairplot.png"))
                    
        except Exception as e:
            results.append((plt.figure(), Markdown(f"❌ Could not generate comparative plots: {e}\n")))
        
        return results

    # Original single-column analysis logic
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
# Enhanced Individual plotting functions
# ------------------------------
def plot_histogram(
    df: pd.DataFrame, 
    col: str, 
    hue: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, Markdown]:
    """Plot a histogram for a numeric column with optional hue grouping."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column '{col}' must be numeric for histogram.")

    fig, ax = (plt.subplots(figsize=(8, 6)) if ax is None else (ax.figure, ax))
    
    if hue and hue in df.columns and df[hue].nunique() <= 10:
        # Grouped histogram
        for group in df[hue].unique():
            group_data = df[df[hue] == group][col].dropna()
            sns.histplot(group_data, kde=True, bins=30, alpha=0.6, 
                        label=str(group), ax=ax)
        ax.legend(title=hue)
    else:
        # Single histogram
        sns.histplot(df[col].dropna(), kde=True, bins=30, color="skyblue", ax=ax)

    ax.set_title(f"Histogram - {col}" + (f" by {hue}" if hue else ""))
    ax.set_xlabel(col)

    md_text = f"## 📈 Histogram: **{col}**"
    if hue:
        md_text += f" by **{hue}**"
    md_text += "\n- Shows distribution of values. Useful for detecting skewness, spread, and modality."
    if hue:
        md_text += f"\n- Grouped by **{hue}** to compare distributions across categories."
    
    return fig, Markdown(md_text)


def plot_boxplot(
    df: pd.DataFrame, 
    col: str, 
    hue: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, Markdown]:
    """Plot a boxplot for a numeric column with optional hue grouping."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise TypeError(f"Column '{col}' must be numeric for boxplot.")

    fig, ax = (plt.subplots(figsize=(8, 6)) if ax is None else (ax.figure, ax))
    
    if hue and hue in df.columns:
        sns.boxplot(data=df, x=hue, y=col, ax=ax)
        ax.set_title(f"Boxplot - {col} by {hue}")
    else:
        sns.boxplot(data=df, y=col, color="lightgreen", ax=ax)
        ax.set_title(f"Boxplot - {col}")

    md_text = f"## 📦 Boxplot: **{col}**"
    if hue:
        md_text += f" by **{hue}**"
    md_text += "\n- Shows spread and outliers. Good for identifying extreme values and comparing distributions."
    if hue:
        md_text += f"\n- Grouped by **{hue}** to compare across categories."
    
    return fig, Markdown(md_text)


def plot_violinplot(
    df: pd.DataFrame, 
    x: str, 
    y: str,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, Markdown]:
    """Plot a violin plot for comparative analysis."""
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Columns '{x}' or '{y}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[y]):
        raise TypeError(f"Column '{y}' must be numeric for violin plot.")

    fig, ax = (plt.subplots(figsize=(10, 6)) if ax is None else (ax.figure, ax))
    sns.violinplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(f"Violin Plot - {y} by {x}")
    ax.tick_params(axis="x", rotation=45 if df[x].nunique() > 5 else 0)

    md = Markdown(
        f"## 🎻 Violin Plot: **{y}** by **{x}**\n"
        f"- Shows distribution of `{y}` across different categories of `{x}`.\n"
        f"- Combines box plot and kernel density estimate for detailed distribution analysis.\n"
    )
    return fig, md


def plot_countplot(
    df: pd.DataFrame, 
    col: str, 
    hue: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, Markdown]:
    """Plot a countplot for a categorical column with optional hue grouping."""
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    fig, ax = (plt.subplots(figsize=(8, 6)) if ax is None else (ax.figure, ax))
    
    if hue and hue in df.columns:
        sns.countplot(data=df, x=col, hue=hue, ax=ax)
        ax.set_title(f"Count Plot - {col} by {hue}")
    else:
        sns.countplot(data=df, x=col, palette="Set2", 
                     order=df[col].value_counts().index, ax=ax)
        ax.set_title(f"Count Plot - {col}")
    
    ax.tick_params(axis="x", rotation=45 if df[col].nunique() > 5 else 0)

    md_text = f"## 🟦 Count Plot: **{col}**"
    if hue:
        md_text += f" by **{hue}**"
    md_text += "\n- Shows frequency of categories. Helpful for understanding class balance."
    if hue:
        md_text += f"\n- Stacked by **{hue}** to show composition within each category."
    
    return fig, Markdown(md_text)


def plot_stacked_barplot(
    df: pd.DataFrame, 
    x: str, 
    hue: str,
    ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, Markdown]:
    """Plot a stacked bar plot for categorical comparisons."""
    if x not in df.columns or hue not in df.columns:
        raise ValueError(f"Columns '{x}' or '{hue}' not found in DataFrame.")

    # Create cross-tabulation for stacked bar plot
    cross_tab = pd.crosstab(df[x], df[hue])
    
    fig, ax = (plt.subplots(figsize=(10, 6)) if ax is None else (ax.figure, ax))
    cross_tab.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f"Stacked Bar Plot - {x} by {hue}")
    ax.set_xlabel(x)
    ax.set_ylabel("Count")
    ax.legend(title=hue)
    ax.tick_params(axis="x", rotation=45 if df[x].nunique() > 5 else 0)

    md = Markdown(
        f"## 📊 Stacked Bar Plot: **{x}** by **{hue}**\n"
        f"- Shows composition of `{hue}` categories within each `{x}` category.\n"
        f"- Useful for understanding proportional relationships between categorical variables.\n"
    )
    return fig, md


def plot_missing_values(df: pd.DataFrame) -> Tuple[plt.Figure, Markdown]:
    """Plot missing values per column."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if missing_counts.empty:
        return plt.figure(), Markdown("✅ No missing values detected in dataset.")

    fig, ax = plt.subplots(figsize=(10, 8))
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


def plot_correlation_matrix(
    df: pd.DataFrame, 
    method: str = 'pearson',
    figsize: Tuple[int, int] = (12, 10)
) -> Tuple[plt.Figure, Markdown]:
    """Plot correlation heatmap and provide insights."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return plt.figure(), Markdown("No numeric columns available for correlation matrix.")

    corr = numeric_df.corr(method=method)
    
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", 
                center=0, square=True, mask=mask, ax=ax)
    ax.set_title(f"Correlation Matrix ({method.title()})")

    # Calculate insights
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
## 🔗 Correlation Matrix ({method.title()})
- Heatmap showing pairwise correlation between numeric features.
- Useful for detecting multicollinearity (features that are too similar).

### 📊 Insights
- **Highest positive correlation:** {max_pos_pair[0]} & {max_pos_pair[1]} → {max_pos_value:.2f}
- **Highest negative correlation:** {max_neg_pair[0]} & {max_neg_pair[1]} → {max_neg_value:.2f}
- **Multicollinearity (|corr| > 0.8):** {multicollinearity_info}
"""
    return fig, Markdown(md_text)


def plot_scatterplot(
    df: pd.DataFrame, 
    x: str, 
    y: str, 
    hue: Optional[str] = None,
    size: Optional[str] = None,
    style: Optional[str] = None,
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (10, 8)
) -> Tuple[plt.Figure, Markdown]:
    """Plot a scatterplot between two numeric variables with advanced options."""
    if x not in df.columns or y not in df.columns:
        raise ValueError("Both x and y must be valid column names in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[x]) or not pd.api.types.is_numeric_dtype(df[y]):
        raise TypeError("Both x and y columns must be numeric for scatterplot.")

    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot with optional dimensions
    scatter_kws = {'alpha': alpha}
    if hue and hue in df.columns:
        scatter_kws['hue'] = df[hue]
    if size and size in df.columns:
        scatter_kws['size'] = df[size]
    if style and style in df.columns:
        scatter_kws['style'] = df[style]
    
    sns.scatterplot(data=df, x=x, y=y, ax=ax, **scatter_kws)
    
    title = f"Scatter Plot: {x} vs {y}"
    if hue:
        title += f" by {hue}"
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    md = [f"## 📊 Scatter Plot: {x} vs {y}"]
    if hue:
        md.append(f"- Colored by **{hue}**")
    if size:
        md.append(f"- Sized by **{size}**")
    if style:
        md.append(f"- Styled by **{style}**")
    md.extend([
        "- Shows relationship between two numeric variables.",
        "- Points may reveal **correlation, clusters, or outliers**."
    ])
    
    return fig, Markdown("\n".join(md))


def plot_pairplot(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None, 
    hue: Optional[str] = None,
    diag_kind: str = "auto",
    corner: bool = False
) -> Tuple[plt.Figure, Markdown]:
    """Plot pairwise relationships across multiple numeric variables."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    
    if columns:
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        plot_data = df[columns]
    else:
        plot_data = df.select_dtypes(include=[np.number])
        if plot_data.empty:
            return plt.figure(), Markdown("No numeric columns available for pairplot.")
        columns = plot_data.columns.tolist()

    # Determine diagonal plot type
    if diag_kind == "auto":
        diag_kind = "kde" if len(plot_data) > 100 else "hist"

    try:
        g = sns.pairplot(
            plot_data,
            hue=hue,
            diag_kind=diag_kind,
            corner=corner,
            palette="viridis",
            plot_kws={'alpha': 0.7}
        )
        title = "Pairplot" + (" (Lower Triangle)" if corner else "")
        g.fig.suptitle(title, y=1.02)

        md = [
            "## 🔍 Pairplot",
            f"- Shows pairwise relationships across {len(columns)} numeric variables.",
            f"- Diagonal shows **{diag_kind} distributions**; off-diagonal shows **scatterplots**.",
            "- Useful for detecting **correlations, clusters, and outliers**."
        ]
        if hue:
            md.append(f"- Data points are colored by **{hue}**.")
        if corner:
            md.append("- Only lower triangle is shown to reduce redundancy.")
            
        return g.fig, Markdown("\n".join(md))

    except Exception as e:
        return plt.figure(), Markdown(f"❌ Could not generate pairplot: {e}")


def plot_facet_grid(
    df: pd.DataFrame,
    x: str,
    y: str,
    col: Optional[str] = None,
    row: Optional[str] = None,
    hue: Optional[str] = None,
    kind: str = "scatter",
    height: int = 4,
    aspect: float = 1.2
) -> Tuple[plt.Figure, Markdown]:
    """Create a facet grid for advanced multivariate analysis."""
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Columns '{x}' or '{y}' not found in DataFrame.")
    
    grid = sns.FacetGrid(df, col=col, row=row, hue=hue, 
                        height=height, aspect=aspect)
    
    if kind == "scatter":
        grid.map(sns.scatterplot, x, y, alpha=0.7)
    elif kind == "line":
        grid.map(sns.lineplot, x, y)
    elif kind == "hist":
        grid.map(sns.histplot, x)
    elif kind == "box":
        grid.map(sns.boxplot, x, y)
    else:
        raise ValueError("Kind must be 'scatter', 'line', 'hist', or 'box'")
    
    grid.add_legend()
    
    title = f"Facet Grid: {y} vs {x}"
    if col:
        title += f" by {col}"
    if row:
        title += f" and {row}"
    if hue:
        title += f" colored by {hue}"
    
    grid.fig.suptitle(title, y=1.02)
    
    md = [f"## 🎯 Facet Grid: {y} vs {x}"]
    if col:
        md.append(f"- Faceted by **{col}**")
    if row:
        md.append(f"- Row facets by **{row}**")
    if hue:
        md.append(f"- Colored by **{hue}**")
    md.append(f"- Plot type: **{kind}**")
    md.append("- Allows detailed comparison across multiple categorical variables.")
    
    return grid.fig, Markdown("\n".join(md))