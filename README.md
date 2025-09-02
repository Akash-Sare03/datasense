# datasense 📊

A Python library for automated exploratory data analysis (EDA), data cleaning, and visualization. Built for for beginners and analysts to quickly understand and preprocess datasets.

---

## Features

- **Dataset Summary**: Overview of shape, dtypes, missing values, duplicates.
- **Missing Value Handling**: Detect and impute missing values (mean, median, mode, constant, drop).
- **Outlier Detection**: Identify outliers using Z-score or IQR methods.
- **Feature Importance**: Calculate and visualize feature importance for regression/classification.
- **Time-Series Analysis**: Decomposition, rolling statistics, and trend detection.
- **Visualizations**: Histograms, boxplots, count plots, correlation matrices, scatter plots, pair plots.
- **Automated Recommendations**: Get actionable insights for data preprocessing.

---

## Installation

```bash
git clone https://github.com/Akash-Sare03/datasense.git
cd datasense
pip install -r requirements.txt
```

---

## Quick Start

```python
import pandas as pd
from datasense import analyze

df = pd.read_csv("your_data.csv")
analyze(df, target_col="price")
```

---

## Usage Examples

### Basic EDA Report
```python
from datasense import analyze

analyze(df, target_col="price", outlier_method="zscore")
```

### Handle Missing Values
```python
from datasense import handle_missing_values

df_clean, report = handle_missing_values(df, method="mean")
```

### Feature Importance
```python
from datasense import feature_importance_calculate

target_type = feature_importance_calculate(df, target_col="target", top_n=10)
```

### Time-Series Analysis
```python
from datasense import analyze_timeseries

analyze_timeseries(df, date_col="date", target_col="sales", freq="D")
```

---

## Example Notebooks

See practical examples and full workflows in the included Jupyter notebooks:

- [Basic EDA Example](notebooks/Datasense_Library_Test_1.ipynb)
- [Time-Series Example](notebooks/Datasense_Library_Test_2.ipynb)

---

## API Reference

### Main Functions
- `analyze()`: Generate a full EDA report.
- `summarize_dataset()`: Dataset overview.
- `handle_missing_values()`: Impute or remove missing values.
- `detect_outliers()`: Find outliers using Z-score or IQR.
- `feature_importance_calculate()`: Compute feature importance.
- `analyze_timeseries()`: Decompose and plot time-series data.
- `visualize()`: Auto-generate plots for numeric/categorical features.

---

## Contributing

1. Fork the project.
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Deployment

This package is available on PyPI. Install it via:

```bash
pip install datasense
```

Or deploy locally:

```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```

---

## Support

If you have any questions or issues, please open an issue [here](https://github.com/Akash-Sare03/datasense/issues).

