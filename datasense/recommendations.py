from __future__ import annotations
from typing import Any
import pandas as pd
import numpy as np
from IPython.display import Markdown


def _to_dataframe(data: Any) -> pd.DataFrame:
    """
    Convert input data into a Pandas DataFrame.

    Args:
        data (Any): Input data. Supported types:
            - pandas.DataFrame → returned as a copy
            - list or dict → converted to DataFrame
            - str → treated as a CSV file path

    Returns:
        pd.DataFrame: Standardized DataFrame.

    Raises:
        TypeError: If input is not a supported type.
        ValueError: If CSV file cannot be read.
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (list, dict)):
        return pd.DataFrame(data)
    if isinstance(data, str):
        try:
            return pd.read_csv(data)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file '{data}': {e}")
    raise TypeError("Unsupported input: use a pandas DataFrame, list, dict, or CSV file path.")


def generate_recommendations(data: Any) -> Markdown:
    """
    Generate actionable, rule-based recommendations for data preprocessing.

    Returns:
        Markdown: A Markdown-formatted recommendations report.
    """
    df = _to_dataframe(data)

    if df.empty:
        raise ValueError("Input DataFrame is empty. Please provide a dataset with rows.")

    report: list[str] = []
    report.append("# 💡 Data Recommendations Report\n")

    # --- 1. Missing values ---
    try:
        missing = df.isnull().sum()
        total_missing = int(missing.sum())
        report.append("## 🚨 Missing Values")
        if total_missing > 0:
            cols_with_missing = missing[missing > 0].index.tolist()
            report.append(
                f"- <span style='color:red; font-weight:bold'>{len(cols_with_missing)} columns</span> have missing values.  "
            )
            report.append(
                "👉 Impute numeric with mean/median, categorical with mode, or drop columns/rows with too many missing values.\n"
            )
            report.append("_Explanation: Handle missing data before ML/analytics_\n")
        else:
            report.append("- <span style='color:green; font-weight:bold'>No missing values ✅</span>\n")
    except Exception as e:
        report.append(f"- ⚠️ Could not calculate missing values due to: {e}\n")

    # --- 2. Columns with too many missing values ---
    try:
        threshold = 0.5
        too_many_missing = [col for col in df.columns if df[col].isnull().mean() > threshold]
        report.append("## 🗑️ High Missing Percentage Columns")
        if too_many_missing:
            report.append(
                f"- Columns **{too_many_missing}** have >50% missing. 👉 Consider dropping them.  "
            )
            report.append("_Explanation: Too many missing values may reduce model reliability_\n")
        else:
            report.append("- No columns with >50% missing values ✅\n")
    except Exception as e:
        report.append(f"- ⚠️ Skipped high-missing analysis due to: {e}\n")

    # --- 3. Constant columns ---
    try:
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
        report.append("## ⚖️ Constant Columns")
        if constant_cols:
            report.append(
                f"- Columns **{constant_cols}** are constant. 👉 Consider dropping them.  "
            )
            report.append("_Explanation: These columns add no information to models_\n")
        else:
            report.append("- No constant columns found ✅\n")
    except Exception as e:
        report.append(f"- ⚠️ Skipped constant column check due to: {e}\n")

    # --- 4. Categorical encoding ---
    try:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        report.append("## 🔤 Categorical Encoding")
        if categorical_cols:
            report.append(
                f"- Dataset contains categorical columns **{categorical_cols}**. "
                "👉 Use Label Encoding (ordinal) or One-Hot Encoding (nominal).\n"
            )
            report.append("_Explanation: ML models need numeric encoding for categorical features_\n")
        else:
            report.append("- No categorical columns found ✅\n")
    except Exception as e:
        report.append(f"- ⚠️ Skipped categorical encoding check due to: {e}\n")

    # --- 5. Skewed numeric columns ---
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skewed = [col for col in numeric_cols if abs(df[col].dropna().skew()) > 1]
        report.append("## 📈 Skewed Features")
        if skewed:
            report.append(
                f"- Columns **{skewed}** are highly skewed. 👉 Consider log transform or normalization.  "
            )
            report.append("_Explanation: Skewed data may hurt algorithms assuming normality_\n")
        else:
            report.append("- No highly skewed columns found ✅\n")
    except Exception as e:
        report.append(f"- ⚠️ Skipped skewness analysis due to: {e}\n")

    # --- 6. Outliers ---
    try:
        report.append("## 🎯 Outliers")
        outlier_cols = []
        for col in numeric_cols:
            if df[col].dropna().empty:
                continue
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:  # avoid div-by-zero issues
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            if not outliers.empty:
                outlier_cols.append(f"{col} ({len(outliers)})")
        if outlier_cols:
            report.append(
                f"- Potential outliers detected in: **{outlier_cols}**. 👉 Consider removal, capping, or transformation.\n"
            )
            report.append("_Explanation: Outliers can skew statistical and ML results_\n")
        else:
            report.append("- No significant outliers detected ✅\n")
    except Exception as e:
        report.append(f"- ⚠️ Skipped outlier detection due to: {e}\n")

    # --- 7. Feature scaling ---
    try:
        report.append("## ⚙️ Feature Scaling")
        if len(numeric_cols) > 0:
            report.append(
                "- Apply StandardScaler or MinMaxScaler if using scale-sensitive algorithms (KNN, SVM, PCA)."
            )
            report.append("_Explanation: Scaling ensures fair contribution of features_\n")
        else:
            report.append("- No numeric columns found to scale ✅\n")
    except Exception as e:
        report.append(f"- ⚠️ Skipped scaling check due to: {e}\n")

    # --- 8. Correlation check ---
    try:
        report.append("## 🔗 Correlation Check")
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            high_corr = [
                (col1, col2)
                for col1 in corr_matrix.columns
                for col2 in corr_matrix.columns
                if col1 != col2 and corr_matrix.loc[col1, col2] > 0.85
            ]
            if high_corr:
                report.append(
                    f"- Highly correlated pairs: **{high_corr}**. 👉 Consider removing one to reduce multicollinearity."
                )
                report.append("_Explanation: Highly correlated features can confuse ML models_\n")
            else:
                report.append("- No highly correlated numeric pairs detected ✅\n")
        else:
            report.append("- Not enough numeric columns for correlation analysis ✅\n")
    except Exception as e:
        report.append(f"- ⚠️ Skipped correlation analysis due to: {e}\n")

    # --- Final fallback ---
    if len(report) == 1:  # only header
        report.append("✅ No major data quality issues detected. Dataset looks clean!")

    return Markdown("\n".join(report))
