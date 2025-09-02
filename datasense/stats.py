import pandas as pd
import numpy as np
from IPython.display import HTML


def calculate_statistics(data: pd.DataFrame, explain: bool = True, columns=None, narrative: bool = True):
    """
    Calculate detailed descriptive statistics for numeric columns
    and return a side-by-side HTML/Markdown report (rendered in Jupyter).

    Args:
        data (pd.DataFrame | str | list | dict): Dataset (DataFrame, CSV path, list, or dict).
        explain (bool): If True, include human-readable explanations.
        narrative (bool): If True, include deeper narrative insights for each column.
        columns (str | list, optional): Column(s) to analyze. 
                                        If None, all numeric columns are analyzed.

    Returns:
        HTML: Rendered HTML report with statistics and optional narrative.
    """

    # --- Convert input to DataFrame ---
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, str):  # CSV file path
        try:
            df = pd.read_csv(data)
        except Exception as e:
            return HTML(f"<p style='color:red;'>❌ Could not read CSV file: {e}</p>")
    elif isinstance(data, (list, dict)):
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            return HTML(f"<p style='color:red;'>❌ Could not convert input to DataFrame: {e}</p>")
    else:
        return HTML("<p style='color:red;'>⚠️ Unsupported input type. Use DataFrame, CSV path, list, or dict.</p>")

    # --- Handle empty dataset ---
    if df.empty:
        return HTML("<p style='color:orange;'>⚠️ The dataset is empty. Nothing to summarize.</p>")

    # --- Handle `columns` parameter ---
    if columns is None:
        selected_cols = df.columns
    elif isinstance(columns, str):
        selected_cols = [columns]  # wrap single column name
    elif isinstance(columns, list):
        selected_cols = columns
    else:
        return HTML("<p style='color:red;'>⚠️ Invalid `columns` parameter. Must be None, str, or list of str.</p>")

    # --- Validate missing columns ---
    missing_cols = [col for col in selected_cols if col not in df.columns]
    if missing_cols:
        return HTML(f"<p style='color:red;'>⚠️ Columns not found: {missing_cols}</p>")

    # --- Subset DataFrame ---
    df = df[selected_cols]

    # --- Select only numeric columns ---
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        return HTML("<p style='color:orange;'>⚠️ No numeric columns found in dataset.</p>")

    # --- Report start ---
    report = []
    report.append("<h1>📊 Detailed Statistics Report</h1>")

    # --- Column-wise statistics ---
    for col in numeric_df.columns:
        try:
            col_data = numeric_df[col].dropna()

            if col_data.empty:
                report.append(f"<h2>🔎 Column: {col}</h2><p style='color:orange;'>⚠️ Column has no valid numeric data.</p><hr>")
                continue

            col_stats = {
                "Count": int(col_data.count()),
                "Missing": int(df[col].isna().sum()),
                "Mean": col_data.mean(),
                "Median": col_data.median(),
                "Mode": col_data.mode().tolist(),
                "Min": col_data.min(),
                "Max": col_data.max(),
                "Range": col_data.max() - col_data.min(),
                "Std Dev": col_data.std(),
                "Variance": col_data.var(),
                "Skewness": col_data.skew(),
                "Kurtosis": col_data.kurt(),
                "25%": col_data.quantile(0.25),
                "50%": col_data.quantile(0.50),
                "75%": col_data.quantile(0.75),
            }

            # Stats table as HTML
            stats_table = pd.DataFrame.from_dict(
                col_stats, orient="index", columns=["Value"]
            ).to_html(border=0, classes="stats-table")

            # --- Explanations ---
            explanations = ""
            if explain:
                explanations += "<ul>"
                explanations += f"<li><b>Mean (Average value)</b> → {col_stats['Mean']:.2f}</li>"
                explanations += f"<li><b>Median (Middle value)</b> → {col_stats['Median']:.2f}</li>"
                explanations += f"<li><b>Mode (Most frequent)</b> → {col_stats['Mode']}</li>"
                explanations += f"<li><b>Range</b> → {col_stats['Min']} to {col_stats['Max']} (spread = {col_stats['Range']:.2f})</li>"
                explanations += f"<li><b>Std Dev (spread of values around mean)</b> → {col_stats['Std Dev']:.2f}</li>"
                explanations += f"<li><b>Variance (spread squared)</b> → {col_stats['Variance']:.2f}</li>"
                explanations += (f"<li><b>Skewness</b> → {col_stats['Skewness']:.2f} "
                                 f"({'Right skew' if col_stats['Skewness'] > 0 else 'Left skew' if col_stats['Skewness'] < 0 else 'Symmetric'})</li>")
                explanations += (f"<li><b>Kurtosis</b> → {col_stats['Kurtosis']:.2f} "
                                 f"({'Heavy tails (outliers present)' if col_stats['Kurtosis'] > 3 else 'Light tails (few outliers)'})</li>")
                explanations += f"<li><b>Percentiles</b> → 25%: {col_stats['25%']:.2f}, 50%: {col_stats['50%']:.2f}, 75%: {col_stats['75%']:.2f}</li>"
                explanations += "</ul>"

            # --- Narrative insights ---
            narrative_block = ""
            if narrative:
                skew = "right-skewed" if col_stats["Skewness"] > 0 else "left-skewed" if col_stats["Skewness"] < 0 else "symmetric"
                variability = "very large" if col_stats["Std Dev"] > (0.75 * abs(col_stats["Mean"])) else "moderate"
                narrative_block = f"""
                <h3>📝 Narrative Insights</h3>
                <p><b>1. Central tendency</b>: Mean = {col_stats['Mean']:.2f}, Median = {col_stats['Median']:.2f}. 
                {'Mean > Median → data is right-skewed.' if col_stats['Mean'] > col_stats['Median'] else 'Mean < Median → data is left-skewed.' if col_stats['Mean'] < col_stats['Median'] else 'Mean ≈ Median → symmetric distribution.'}</p>
                
                <p><b>2. Spread / variability</b>: Std Dev = {col_stats['Std Dev']:.2f}, which is {variability} compared to the mean.</p>

                <p><b>3. Range</b>: {col_stats['Range']:.2f} (Min = {col_stats['Min']}, Max = {col_stats['Max']}).</p>

                <p><b>4. Possible shape</b>: Distribution appears {skew}. 
                Kurtosis = {col_stats['Kurtosis']:.2f}, suggesting {'outliers/heavy tails' if col_stats['Kurtosis'] > 3 else 'light tails/few outliers'}.</p>

                <p><b>5. Takeaways</b>: 
                {'Potential outliers detected, consider winsorization or transformation.' if col_stats['Kurtosis'] > 3 else 'Data looks relatively stable.'}
                Scaling/normalization may be required before modeling.</p>
                """

            # --- Side-by-side layout ---
            section = f"""
            <h2>🔎 Column: {col}</h2>
            <div style="display:flex; gap:30px; align-items:flex-start; margin-bottom:20px;">
                <div style="flex:1; min-width:300px;">{stats_table}</div>
                <div style="flex:1; min-width:300px;">{explanations}{narrative_block}</div>
            </div>
            <hr>
            """
            report.append(section)

        except Exception as e:
            report.append(f"<h2>🔎 Column: {col}</h2><p style='color:red;'>⚠️ Error analyzing column: {e}</p><hr>")

    return HTML("\n".join(report))
