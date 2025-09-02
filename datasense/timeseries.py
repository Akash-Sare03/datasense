import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from IPython.display import display, Markdown

def analyze_timeseries(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    freq: str = "D",
    window: int = 7
):
    """
    Beginner-friendly time-series analysis with explanations + auto-insights.
    
    Args:
        df (pd.DataFrame): Input dataset.
        date_col (str): Column with datetime values.
        target_col (str): Column with numeric values to analyze.
        freq (str): Frequency for resampling (e.g., 'D', 'M').
        window (int): Rolling window size for stats.
    """
    # -------------------------------
    # 0. Input validation
    # -------------------------------
    if df.empty:
        display(Markdown("⚠️ Provided DataFrame is **empty**. Nothing to analyze."))
        return
    
    if date_col not in df.columns or target_col not in df.columns:
        display(Markdown(f"❌ Columns `{date_col}` or `{target_col}` not found in DataFrame."))
        return

    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    except Exception as e:
        display(Markdown(f"❌ Could not convert `{date_col}` to datetime: {e}"))
        return

    if df[date_col].isna().all():
        display(Markdown(f"❌ Column `{date_col}` could not be parsed as valid datetimes."))
        return

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        display(Markdown(f"❌ Column `{target_col}` must be numeric, but got {df[target_col].dtype}."))
        return

    # -------------------------------
    # 1. Prepare time series
    # -------------------------------
    ts = df.set_index(date_col)[target_col].asfreq(freq)

    if ts.dropna().empty:
        display(Markdown(f"⚠️ Target column `{target_col}` has no valid numeric values after processing."))
        return

    display(Markdown("## ⏳ Time-Series Analysis"))

    insights = []  # collect insights for final report

    # -------------------------------
    # 2. Rolling Mean & Std
    # -------------------------------
    try:
        rolling_mean = ts.rolling(window=window).mean()
        rolling_std = ts.rolling(window=window).std()

        fig, ax = plt.subplots(figsize=(10, 5))
        ts.plot(ax=ax, color="blue", alpha=0.6, label="Original")
        rolling_mean.plot(ax=ax, color="red", label=f"Rolling Mean ({window})")
        rolling_std.plot(ax=ax, color="green", label=f"Rolling Std ({window})")
        ax.legend()
        ax.set_title("Rolling Mean & Standard Deviation")
        plt.show(fig)

        # Insight for trend
        if len(rolling_mean.dropna()) >= 2:
            if rolling_mean.dropna().iloc[-1] > rolling_mean.dropna().iloc[0]:
                trend_msg = "📈 The rolling mean suggests an **increasing trend** over time."
            elif rolling_mean.dropna().iloc[-1] < rolling_mean.dropna().iloc[0]:
                trend_msg = "📉 The rolling mean suggests a **decreasing trend**."
            else:
                trend_msg = "➖ The rolling mean suggests a **stable/flat trend**."
        else:
            trend_msg = "ℹ️ Not enough data points to determine trend."

        display(Markdown(trend_msg))
        insights.append(trend_msg)

        # Insight for volatility
        if rolling_std.mean() > 0.5 * ts.std():
            vol_msg = "⚠️ The series shows **high volatility** (large fluctuations)."
        else:
            vol_msg = "✅ The series has **low/moderate volatility**."
        display(Markdown(vol_msg))
        insights.append(vol_msg)

    except Exception as e:
        err_msg = f"⚠️ Could not compute rolling statistics: {e}"
        display(Markdown(err_msg))
        insights.append(err_msg)

    # -------------------------------
    # 3. Seasonal Decomposition
    # -------------------------------
    try:
        decomposition = seasonal_decompose(ts.dropna(), model="additive", period=None)
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        plt.show(fig)

        display(Markdown("🔍 **Decomposition Insights:**"))
        decomp_msg = [
            "- **Trend** shows the long-term direction of the series.",
            "- **Seasonal** captures repeated patterns (e.g., daily, weekly, yearly).",
            "- **Residual** highlights irregular fluctuations not explained by trend/seasonality."
        ]
        for msg in decomp_msg:
            display(Markdown(msg))
        insights.extend(decomp_msg)

        # Auto-detect seasonality strength
        if decomposition.seasonal.std() > 0.1 * ts.std():
            seas_msg = "🌊 The decomposition reveals **strong seasonality** in the data."
        else:
            seas_msg = "📏 The decomposition shows **weak or minimal seasonality**."
        display(Markdown(seas_msg))
        insights.append(seas_msg)

    except Exception as e:
        err_msg = f"⚠️ Could not perform seasonal decomposition: {e}"
        display(Markdown(err_msg))
        insights.append(err_msg)

    # -------------------------------
    # 4. Final Summary Report
    # -------------------------------
    display(Markdown("## 📝 Summary Report"))
    for i, msg in enumerate(insights, 1):
        display(Markdown(f"{i}. {msg}"))
