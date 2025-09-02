from __future__ import annotations
from typing import Any, Optional, Union
import pandas as pd
from IPython.display import display, Markdown


def _to_dataframe(data: Any) -> pd.DataFrame:
    """
    Convert input data into a Pandas DataFrame.

    Args:
        data (Any): Input dataset. Supported formats:
            - pandas.DataFrame → returned as a copy
            - list or dict → converted to DataFrame
            - str → treated as a CSV file path

    Returns:
        pd.DataFrame: Standardized DataFrame.

    Raises:
        TypeError: If the input type is not supported.
        FileNotFoundError: If CSV path is invalid.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (list, dict)):
        return pd.DataFrame(data)
    if isinstance(data, str):
        try:
            return pd.read_csv(data)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"CSV file not found: {data}") from e
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}") from e
    raise TypeError("Unsupported input: use a pandas DataFrame, list, dict, or CSV file path.")


def find_missing_values(data: Any) -> Markdown:
    """
    Generate a Markdown report summarizing missing values in the dataset.

    Returns friendly warnings for empty datasets.
    """
    df = _to_dataframe(data)

    if df.empty:
        return Markdown("⚠️ <span style='color:orange; font-weight:bold'>The dataset is empty. No missing values to analyze.</span>")

    report: list[str] = []
    report.append("## 🚨 Missing Values Report 🚨\n")
    report.append("---")

    try:
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
    except Exception as e:
        return Markdown(f"<p style='color:red;'>⚠️ Error while calculating missing values: {e}</p>")

    missing_df = pd.DataFrame({
        "Missing Values": missing_count,
        "Percentage": missing_percent
    })
    missing_df = missing_df[missing_df["Missing Values"] > 0].sort_values(
        by="Missing Values", ascending=False
    )

    if missing_df.empty:
        report.append("- <span style='color:green; font-weight:bold'>✅ No missing values found</span>\n")
        report.append("_Explanation: Every column is complete with no missing data._\n")
    else:
        total_missing = int(missing_count.sum())
        report.append(f"- **Total Missing Values**: <span style='color:red; font-weight:bold'>{total_missing}</span> ❌\n")

        # Show top 5 missing columns
        top_missing = missing_df.head(5)
        top_missing_str = ", ".join(
            [f"`{col}` ({row['Missing Values']} → {row['Percentage']:.2f}%)"
             for col, row in top_missing.iterrows()]
        )
        report.append(f"- Columns with most missing: {top_missing_str}\n")
        report.append("_Explanation: Shows the columns with the largest number of missing entries._\n")

        # Full missing value table
        report.append("### 📋 Detailed Missing Value Table")
        report.append(missing_df.to_markdown())
        report.append("_Explanation: Count and percentage of missing values per column._\n")

    return Markdown("\n".join(report))


class MissingValueResult:
    """
    Wrapper class for missing value handling results.

    Attributes:
        df (pd.DataFrame): Cleaned DataFrame after missing value handling.
        report (Markdown): Markdown-formatted report of the applied method.
    """

    def __init__(self, df: pd.DataFrame, report: Markdown):
        self.df = df
        self.report = report

    def _repr_markdown_(self) -> str:
        """Render Markdown report in Jupyter automatically."""
        return str(self.report.data)


def handle_missing_values(
    data: Any,
    method: str = "drop",
    value: Optional[Union[int, float, str]] = None
) -> tuple[pd.DataFrame, Markdown]:
    """
    Handle missing values in a dataset according to the chosen strategy.

    Raises friendly errors for invalid method or empty datasets.
    """
    df = _to_dataframe(data)

    if df.empty:
        return df, Markdown("⚠️ <span style='color:orange; font-weight:bold'>The dataset is empty. No missing values to handle.</span>")

    df_copy = df.copy()
    report: list[str] = ["## 🛠️ Missing Value Handling 🛠️\n"]
    display(Markdown("---"))


    try:
        if method == "drop":
            df_copy = df_copy.dropna()
            report.append("- 🗑️ Rows containing missing values were **dropped**.\n")

        elif method == "mean":
            numeric_cols = df_copy.select_dtypes(include=["float64", "int64"]).columns
            if numeric_cols.empty:
                return df, Markdown("⚠️ <span style='color:orange;'>No numeric columns available for mean imputation.</span>")
            for col in numeric_cols:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            report.append("- 📊 Missing numeric values replaced with **column means**.\n")

        elif method == "median":
            numeric_cols = df_copy.select_dtypes(include=["float64", "int64"]).columns
            if numeric_cols.empty:
                return df, Markdown("⚠️ <span style='color:orange;'>No numeric columns available for median imputation.</span>")
            for col in numeric_cols:
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            report.append("- 📊 Missing numeric values replaced with **column medians**.\n")

        elif method == "mode":
            for col in df_copy.columns:
                mode_val = df_copy[col].mode()
                if not mode_val.empty:
                    df_copy[col] = df_copy[col].fillna(mode_val[0])
            report.append("- 📊 Missing values replaced with **most frequent value (mode)** of each column.\n")

        elif method == "constant":
            if value is None:
                raise ValueError("For method='constant', you must provide a value.")
            df_copy = df_copy.fillna(value)
            report.append(f"- 🧱 All missing values replaced with constant value: **{value}**\n")

        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'drop', 'mean', 'median', 'mode', or 'constant'.")

    except Exception as e:
        return df, Markdown(f"<p style='color:red;'>⚠️ Error while handling missing values: {e}</p>")

    return df_copy, Markdown("\n".join(report))
