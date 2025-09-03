# datasense/__init__.py

# --- Version Handling (auto from package metadata) ---
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # for Python <3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("datasense")
except PackageNotFoundError:
    __version__ = "unknown"


# --- Public API Imports ---
from .summary import summarize_dataset
from .stats import calculate_statistics
from .recommendations import generate_recommendations
from .outliers import detect_outliers, remove_outliers
from .missing_values import find_missing_values, handle_missing_values
from .feature_importance import feature_importance_calculate
from .analyze_data import analyze
from .timeseries import analyze_timeseries
from .visualization import (
    visualize,
    plot_missing_values,
    plot_histogram,
    plot_boxplot,
    plot_countplot,
    plot_correlation_matrix,
    plot_scatterplot,
    plot_pairplot,
)

# --- Public API List ---
__all__ = [
    "summarize_dataset",
    "calculate_statistics",
    "generate_recommendations",
    "detect_outliers",
    "remove_outliers",
    "find_missing_values",
    "handle_missing_values",
    "feature_importance_calculate",
    "analyze",
    "analyze_timeseries",
    "visualize",
    "plot_missing_values",
    "plot_histogram",
    "plot_boxplot",
    "plot_countplot",
    "plot_correlation_matrix",
    "plot_scatterplot",
    "plot_pairplot",
]
