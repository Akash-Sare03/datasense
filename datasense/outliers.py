# outliers.py

from __future__ import annotations
from typing import Any, Tuple, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from scipy.stats import zscore


def _to_dataframe(data: Any) -> pd.DataFrame:
    """
    Convert input data into a pandas DataFrame.

    Raises
    ------
    TypeError : If input type is unsupported.
    FileNotFoundError : If CSV path is invalid.
    ValueError : If CSV cannot be read.
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


def detect_outliers(
    data: Any,
    method: str = 'zscore',
    threshold: float = 3.0,
    sample_rows: int = 5,
    visualize: bool = True
) -> Union[Markdown, Tuple[Markdown, dict[str, plt.Figure]]]:
    """
    Detect outliers in numeric columns using Z-score or IQR.
    Returns a Markdown report and optional visualizations.
    """

    df = _to_dataframe(data)

    if df.empty:
        return Markdown("⚠️ <span style='color:orange;'>The dataset is empty. No outliers to detect.</span>")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return Markdown("⚠️ **No numeric columns found in the dataset.**")

    outlier_flags = pd.DataFrame(False, index=df.index, columns=numeric_df.columns)
    explanations = []

    try:
        if method.lower() == 'zscore':
            z_scores = (numeric_df - numeric_df.mean()) / numeric_df.std(ddof=0)
            outlier_flags = abs(z_scores) > threshold
            for col in numeric_df.columns:
                mean = numeric_df[col].mean()
                std = numeric_df[col].std()
                count = outlier_flags[col].sum()
                explanations.append(
                    f"- **{col}**: {count} outliers detected using Z-score (threshold={threshold}). "
                    f"Mean={mean:.2f}, Std={std:.2f}"
                )

        elif method.lower() == 'iqr':
            for col in numeric_df.columns:
                Q1 = numeric_df[col].quantile(0.25)
                Q3 = numeric_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_flags[col] = (numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)

                count = outlier_flags[col].sum()
                explanations.append(
                    f"- **{col}**: {count} outliers detected using IQR. "
                    f"Range=[{lower_bound:.2f}, {upper_bound:.2f}] (Q1={Q1:.2f}, Q3={Q3:.2f})"
                )
        else:
            raise ValueError("Invalid method. Use 'zscore' or 'iqr'.")

    except Exception as e:
        return Markdown(f"<p style='color:red;'>⚠️ Error while detecting outliers: {e}</p>")

    # Summary table
    summary_df = pd.DataFrame({
        "Outlier Count": outlier_flags.sum(),
        "Outlier %": (outlier_flags.sum() / len(df) * 100).round(2)
    })

    outlier_rows = df[outlier_flags.any(axis=1)]

    # Build Markdown report
    report = []
    report.append("## 🚨 Outlier Detection Report 🚨\n")
    report.append("---")
    if method.lower() == 'zscore':
        report.append("### 📐 Z-Score Method Explanation")
        report.append(f"A threshold of `{threshold}` means values more than {threshold} standard deviations away are considered outliers.  \n")
    elif method.lower() == 'iqr':
        report.append("### 📊 IQR Method Explanation")
        report.append("Any value below `Q1 - 1.5 × IQR` or above `Q3 + 1.5 × IQR` is flagged as an outlier.  \n")

    report.append(f"**Method Used**: `{method}` (Threshold={threshold})\n")
    report.append("### 📊 Summary Table")
    report.append(summary_df.to_markdown())
    report.append("### 📝 Column-wise Explanations")
    report.extend(explanations)

    if outlier_rows.empty:
        report.append("✅ No outliers detected in the dataset.\n")

    md_report = Markdown("\n".join(report))

    # Visualization (safe execution)
    figures: dict[str, plt.Figure] = {}
    if visualize:
        try:
            for col in numeric_df.columns:
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                sns.boxplot(x=numeric_df[col], ax=axes[0])
                axes[0].set_title(f"{col} - Boxplot")
                sns.histplot(numeric_df[col], bins=30, kde=True, ax=axes[1])
                axes[1].axvline(numeric_df[col].mean(), color='green', linestyle='--', label='Mean')
                axes[1].set_title(f"{col} - Histogram")
                axes[1].legend()
                figures[col] = fig
        except Exception as e:
            return Markdown(f"<p style='color:red;'>⚠️ Error while generating visualizations: {e}</p>")

        return md_report, figures

    return md_report


def remove_outliers(
    data: Any,
    method: str = 'iqr',
    multiplier: float = 1.5,
    threshold: float = 3.0,
    strategy: str = "remove"
) -> Tuple[pd.DataFrame, Markdown]:
    """
    Handle outliers in dataset using IQR or Z-score.
    Returns friendly errors for invalid methods or empty datasets.
    """

    df = _to_dataframe(data)
    if df.empty:
        return df, Markdown("⚠️ <span style='color:orange;'>The dataset is empty. No outliers to remove.</span>")

    df_clean = df.copy()
    explanation = ["## 🧹 Outlier Handling Report\n"]
    display(Markdown("---"))

    try:
        if method.lower() == 'iqr':
            for col in df_clean.select_dtypes(include='number').columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR

                if strategy == "remove":
                    before = len(df_clean)
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
                    after = len(df_clean)
                    explanation.append(f"- **{col}**: Removed {before - after} rows (IQR range = [{lower_bound:.2f}, {upper_bound:.2f}])")
                elif strategy == "cap":
                    outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
                    explanation.append(f"- **{col}**: Capped {outliers} values to IQR range [{lower_bound:.2f}, {upper_bound:.2f}]")
                elif strategy == "nan":
                    outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                    df_clean.loc[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), col] = np.nan
                    explanation.append(f"- **{col}**: Replaced {outliers} outliers with NaN (IQR range = [{lower_bound:.2f}, {upper_bound:.2f}])")
                else:
                    raise ValueError(f"Invalid strategy: {strategy}. Use 'remove', 'cap', or 'nan'.")

        elif method.lower() == 'zscore':
            for col in df_clean.select_dtypes(include='number').columns:
                z_scores = zscore(df_clean[col].dropna())
                mask = abs(z_scores) < threshold
                if strategy == "remove":
                    before = len(df_clean)
                    df_clean = df_clean[mask.reindex(df_clean.index, fill_value=False)]
                    after = len(df_clean)
                    explanation.append(f"- **{col}**: Removed {before - after} rows (Z-score threshold={threshold})")
                elif strategy == "cap":
                    mean, std = df_clean[col].mean(), df_clean[col].std()
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                    df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
                    explanation.append(f"- **{col}**: Capped {outliers} values to Z-score bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
                elif strategy == "nan":
                    z_scores_full = abs(zscore(df_clean[col].dropna()))
                    mask_full = z_scores_full >= threshold
                    outliers = mask_full.sum()
                    df_clean.loc[mask_full.index[mask_full], col] = np.nan
                    explanation.append(f"- **{col}**: Replaced {outliers} outliers with NaN (Z-score threshold={threshold})")
                else:
                    raise ValueError(f"Invalid strategy: {strategy}. Use 'remove', 'cap', or 'nan'.")
        else:
            raise ValueError("Invalid method. Use 'iqr' or 'zscore'.")

    except Exception as e:
        return df, Markdown(f"<p style='color:red;'>⚠️ Error while removing outliers: {e}</p>")

    explanation.append("\n_Explanation: Outliers handled using chosen method & strategy_")
    return df_clean, Markdown("\n".join(explanation))
