from __future__ import annotations
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from IPython.display import display, Markdown


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
        TypeError: If input type is not supported.
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


def feature_importance_calculate(
    data: Any,
    target_col: str,
    top_n: int = 10,
    show_bottom: bool = False
) -> tuple[pd.DataFrame, str, str]:
    """
    Calculate and visualize feature importance for a dataset.

    Returns:
        - pd.DataFrame: Feature importance sorted descending
        - str: Markdown report string
        - str: Target type ("regression" | "classification")
    """
    df = _to_dataframe(data)

    if df.empty:
        raise ValueError("Input DataFrame is empty. Please provide a dataset with rows.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    if df.empty:
        raise ValueError(f"All rows have missing values in target column '{target_col}'.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if X.empty:
        raise ValueError("No features available after dropping target column.")

    importances: dict[str, float] = {}

    # --- Detect target type ---
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() <= 20:  
            target_type = "classification"
        else:
            target_type = "regression"
    else:
        target_type = "classification"

    # --- REGRESSION ---
    if target_type == "regression":
        numeric_features = X.select_dtypes(include=[np.number])
        if numeric_features.empty:
            raise ValueError("No numeric features available for regression importance calculation.")

        for col in numeric_features.columns:
            corr = df[col].corr(df[target_col])
            if pd.notna(corr):
                importances[col] = abs(corr)

        cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if cat_features:
            try:
                X_encoded = pd.get_dummies(X[cat_features], drop_first=True)
                mi_scores = mutual_info_regression(X_encoded, y, random_state=42)
                for col, score in zip(X_encoded.columns, mi_scores):
                    importances[col] = score
            except Exception as e:
                print(f"⚠️ Warning: Failed to calculate mutual info for categorical features: {e}")

    # --- CLASSIFICATION ---
    else:
        try:
            X_encoded = pd.get_dummies(X, drop_first=True)
            if X_encoded.empty:
                raise ValueError("After encoding, no valid features found for classification.")
            y_encoded = y.astype("category").cat.codes
            mi_scores = mutual_info_classif(X_encoded, y_encoded, random_state=42)
            for col, score in zip(X_encoded.columns, mi_scores):
                importances[col] = score
        except Exception as e:
            raise RuntimeError(f"Error calculating feature importance for classification: {e}")

    # Create DataFrame
    importance_df = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"])
    importance_df.sort_values(by="Importance", ascending=False, inplace=True)

    # --- PLOT SAFELY ---
    if not importance_df.empty:
        try:
            plt.figure(figsize=(8, 6))
            sns.barplot(
                x="Importance", 
                y="Feature", 
                data=importance_df.head(top_n), 
                palette="viridis"
            )
            plt.title(f"Top {top_n} Feature Importances ({target_type.title()})")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.show()

            if show_bottom:
                plt.figure(figsize=(8, 6))
                sns.barplot(
                    x="Importance", 
                    y="Feature", 
                    data=importance_df.tail(top_n), 
                    palette="magma"
                )
                plt.title(f"Bottom {top_n} Feature Importances ({target_type.title()})")
                plt.xlabel("Importance")
                plt.ylabel("Feature")
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"⚠️ Visualization skipped due to error: {e}")

    # --- MARKDOWN REPORT ---
    report: list[str] = []
    report.append("# 🔍 Feature Importance Report\n")
    report.append(f"- Target column: **`{target_col}`**\n")
    report.append(f"- Detected task type: **{target_type.title()}**\n")
    report.append(f"- Total features evaluated: **{len(importances)}**\n")

    if importance_df.empty:
        report.append("<span style='color:red; font-weight:bold'>⚠️ No valid features found for importance calculation.</span>\n")
    else:
        top_features = importance_df.head(top_n)
        top_str = ", ".join(
            [f"`{row['Feature']}` ({row['Importance']:.3f})"
             for _, row in top_features.iterrows()]
        )
        report.append(f"- Top {top_n} features: {top_str}\n")

        if show_bottom:
            bottom_features = importance_df.tail(top_n)
            bottom_str = ", ".join(
                [f"`{row['Feature']}` ({row['Importance']:.3f})"
                 for _, row in bottom_features.iterrows()]
            )
            report.append(f"- Bottom {top_n} features: {bottom_str}\n")

        report.append("\n## 📊 Full Importance Table\n")
        report.append(importance_df.to_markdown(index=False))

    md_report: str = "\n".join(report)

    # Auto display in Jupyter
    display(Markdown(md_report))

    return target_type
