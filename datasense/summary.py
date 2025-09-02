from __future__ import annotations
from typing import Any
import pandas as pd
import numpy as np
from IPython.display import display, Markdown


def _to_dataframe(data: Any) -> pd.DataFrame:
    """
    Convert various input types into a pandas DataFrame.

    Args:
        data (Any): Input data. Can be:
            - pandas DataFrame (returns a copy),
            - list or dict (converted to DataFrame),
            - str (interpreted as a CSV file path).

    Returns:
        pd.DataFrame: Standardized DataFrame.

    Raises:
        TypeError: If the input type is unsupported.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (list, dict)):
        return pd.DataFrame(data)
    if isinstance(data, str):
        try:
            return pd.read_csv(data)
        except Exception as e:
            raise ValueError(f"❌ Could not read CSV file '{data}': {e}")
    raise TypeError("❌ Unsupported input: use a pandas DataFrame, list, dict, or CSV file path.")


def summarize_dataset(
    data: Any,
    include_sample: bool = True,
    sample_rows: int = 5,
) -> Markdown:
    """
    Generate a detailed Markdown summary report of a dataset.

    Args:
        data (Any): Input data (DataFrame, list, dict, or CSV path).
        include_sample (bool, optional): Whether to include sample rows. Defaults to True.
        sample_rows (int, optional): Number of rows to show. Defaults to 5.

    Returns:
        Markdown: Renderable Markdown report (for Jupyter Notebooks).
    """
    try:
        df = _to_dataframe(data)
    except Exception as e:
        return Markdown(f"⚠️ **Error:** Could not convert input to DataFrame. Details: {e}")

    if df.empty:
        return Markdown("⚠️ The dataset is **empty**. Nothing to summarize.")

    report: list[str] = []
    report.append("## 📊 Dataset Summary Report 📊\n")
    report.append("---")

    # ===== Shape =====
    try:
        report.append("### 📝 Shape")
        report.append(f"- **{df.shape[0]} rows, {df.shape[1]} columns**  ")
        report.append("_Explanation: Number of rows and columns in the dataset_\n")
    except Exception as e:
        report.append(f"⚠️ Could not determine shape: {e}\n")

    # ===== Column dtypes =====
    try:
        report.append("### 🔤 Column Data Types")
        dtype_summary = ", ".join([f"`{col}`: {dtype}" for col, dtype in df.dtypes.astype(str).items()])
        report.append(f"- {dtype_summary}  ")
        report.append("_Explanation: Data types of each column_\n")
    except Exception as e:
        report.append(f"⚠️ Could not determine column types: {e}\n")

    # ===== Missing values =====
    try:
        miss_count = df.isna().sum()
        total_missing = int(miss_count.sum())
        report.append("### 🚨 Missing Values")
        if total_missing > 0:
            top_missing = miss_count.sort_values(ascending=False).head(3)
            missing_str = ", ".join([f"`{col}` ({cnt})" for col, cnt in top_missing.items() if cnt > 0])
            report.append(f"- **Total Missing**: <span style='color:red; font-weight:bold'>{total_missing}</span> ❌  ")
            report.append(f"- Most missing in: {missing_str}  ")
            report.append("_Explanation: Shows how much data is missing across columns_\n")
        else:
            report.append("- <span style='color:green; font-weight:bold'>No missing values ✅</span>  ")
            report.append("_Explanation: There are no missing values in the dataset_\n")
    except Exception as e:
        report.append(f"⚠️ Could not analyze missing values: {e}\n")

    # ===== Duplicate rows =====
    try:
        dup_count = df.duplicated().sum()
        report.append("### 🔁 Duplicate Rows")
        if dup_count > 0:
            report.append(f"- <span style='color:orange; font-weight:bold'>{dup_count} duplicate rows detected ⚠️</span>  ")
        else:
            report.append("- <span style='color:green; font-weight:bold'>No duplicate rows ✅</span>  ")
        report.append("_Explanation: Number of identical rows in the dataset_\n")
    except Exception as e:
        report.append(f"⚠️ Could not check duplicates: {e}\n")

    # ===== Unique counts =====
    try:
        report.append("### 🔑 Unique Values")
        unique_counts = df.nunique(dropna=True)
        if not unique_counts.empty:
            top_unique = unique_counts.sort_values(ascending=False).head(3)
            unique_str = ", ".join([f"`{col}` ({cnt})" for col, cnt in top_unique.items()])
            report.append(f"- Top unique count columns: {unique_str}  ")
            report.append("_Explanation: Columns with the highest number of unique values_\n")
        else:
            report.append("- No unique values detected.\n")
    except Exception as e:
        report.append(f"⚠️ Could not compute unique values: {e}\n")

    # ===== Numeric stats =====
    try:
        numeric_df = df.select_dtypes(include="number")
        report.append("### 🔢 Numeric Summary")
        if not numeric_df.empty:
            desc = numeric_df.describe().T
            desc["median"] = numeric_df.median()
            desc = desc[["mean", "median", "std", "min", "max"]].round(3)

            report.append(desc.to_markdown())

            # Add explanations
            report.append("\n### ℹ️ Explanation of Statistics")
            report.append("- **Mean** → Average value of the column")
            report.append("- **Median** → Middle value (less affected by outliers)")
            report.append("- **Std Dev** → Spread of values (higher = more variation)")
            report.append("- **Min** → Lowest value in the column")
            report.append("- **Max** → Highest value in the column\n")
        else:
            report.append("- No numeric columns available  ")
            report.append("_Explanation: Dataset contains no numeric columns_\n")
    except Exception as e:
        report.append(f"⚠️ Could not compute numeric summary: {e}\n")

    # ===== Sample rows =====
    if include_sample:
        try:
            report.append("### 📌 Sample Rows")
            sample_str = df.head(sample_rows).to_markdown(index=False)
            report.append(sample_str)
            report.append(f"_Explanation: First {sample_rows} rows of the dataset_\n")
        except Exception as e:
            report.append(f"⚠️ Could not display sample rows: {e}\n")

    return Markdown("\n".join(report))
