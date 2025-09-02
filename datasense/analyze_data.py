import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from typing import Union, List, Optional

from datasense.summary import summarize_dataset
from datasense.missing_values import find_missing_values
from datasense.visualization import plot_missing_values
from datasense.outliers import detect_outliers


def analyze(
    df: pd.DataFrame, 
    target_col: Optional[Union[str, List[str]]] = None, 
    outlier_method: str = "zscore"
) -> None:
    """
    Generate a Markdown-based EDA report with inline plots.
    
    Args:
        df (pd.DataFrame): Input dataset.
        target_col (Union[str, List[str], None]): Target column name(s). 
            If None, analyzes full dataframe. If provided, focuses on specific column(s).
        outlier_method (str): Method for outlier detection ('zscore' or 'iqr').
    
    Returns:
        None – Displays Markdown + Plots directly (Jupyter/Streamlit friendly).
    """

    # ====== 🔹 Basic Validation ======
    if df is None or df.empty:
        display(Markdown("⚠️ **Warning:** The dataset is empty. Nothing to analyze."))
        return

    display(Markdown("# 🔍📑 Detailed Exploration of Data, Missing Values and Outliers 📑🔍"))
    display(Markdown("---"))


    # Determine which columns to analyze
    if target_col is not None:
        # Convert single column to list for consistent processing
        if isinstance(target_col, str):
            target_columns = [target_col]
        else:
            target_columns = target_col
        
        # Validate target columns exist
        invalid_columns = [col for col in target_columns if col not in df.columns]
        if invalid_columns:
            display(Markdown(f"❌ **Error:** Target column(s) `{invalid_columns}` not found in dataset."))
            display(Markdown("---"))

            return
        
        analysis_df = df[target_columns]
        display(Markdown(f"## 🎯 Focused Analysis on: `{target_columns}`"))
        display(Markdown("---"))
        
    else:
        analysis_df = df

    # ====== 1. Dataset Overview ======
    try:
        summary = summarize_dataset(analysis_df)
        display(summary)
        display(Markdown("---"))
    except Exception as e:
        display(Markdown(f"⚠️ **Error generating dataset summary:** {str(e)}"))
        display(Markdown("---"))

    # ====== 2. Missing Values ======
    try:
        mv_report = find_missing_values(analysis_df)
        display(mv_report)
        display(Markdown("---"))

        fig, mv_plot = plot_missing_values(analysis_df)
        if fig:
            display(mv_plot)
            plt.show(fig)
            display(Markdown("---"))
    except Exception as e:
        display(Markdown(f"⚠️ **Error analyzing missing values:** {str(e)}"))
        display(Markdown("---"))

    # ====== 3. Outlier Detection ======
    try:
        # Only analyze numeric columns for outliers
        numeric_columns = analysis_df.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_columns:
            outlier_result = detect_outliers(
                analysis_df[numeric_columns], 
                method=outlier_method, 
                visualize=True
            )

            if isinstance(outlier_result, tuple):
                md_report, figures = outlier_result
                display(md_report)
                display(Markdown("---"))
                for col, fig in figures.items():
                    try:
                        plt.show(fig)
                    except Exception as plot_err:
                        display(Markdown(f"⚠️ Could not plot outliers for `{col}`: {plot_err}"))
            else:
                display(outlier_result)
                display(Markdown("---"))

        else:
            display(Markdown("ℹ️ **Info:** No numeric columns found for outlier analysis."))
            display(Markdown("---"))
    except Exception as e:
        display(Markdown(f"⚠️ **Error during outlier detection:** {str(e)}"))
        display(Markdown("---"))

    display(Markdown("✅ **Report generation complete.**"))