# datasense 📊

[![Python Tests](https://github.com/Akash-Sare03/datasense/actions/workflows/python-test.yml/badge.svg)](https://github.com/Akash-Sare03/datasense/actions/workflows/python-test.yml)

[[Documentation](https://github.com/Akash-Sare03/datasense/blob/main/documentation.md)]


A Python library for automated exploratory data analysis (EDA), data cleaning, and visualization. 
Built for beginners and analysts to quickly understand and preprocess datasets.

---

## ✨ Features

- **Dataset Summary**: Overview of shape, dtypes, missing values, duplicates.  
- **Missing Value Handling**: Detect and impute missing values (mean, median, mode, constant, drop).  
- **Outlier Detection**: Identify outliers using Z-score or IQR methods.  
- **Feature Importance**: Calculate and visualize feature importance for regression/classification.  
- **Time-Series Analysis**: Decomposition, rolling statistics, and trend detection.  
- **Visualizations**: Histograms, boxplots, count plots, correlation matrices, scatter plots, pair plots.  
- **Automated Recommendations**: Get actionable insights for data preprocessing.  

---

## 🚀 Why Datasense?

- ⚡ One-line automated EDA for quick dataset understanding.  
- 🧹 Built-in cleaning and preprocessing to save time.  
- 📊 Visual + tabular insights, ready for analysis or ML pipelines.  
- 🔧 Beginner-friendly but powerful enough for production workflows.  

---

## Installation

```bash
git clone https://github.com/Akash-Sare03/datasense.git
cd datasense
pip install -r requirements.txt

Or install from PyPI: 
pip install datasense-eda
```
---

## Quick Start

```python
import pandas as pd
from datasense import analyze

df = pd.read_csv("your_data.csv")

# Run a full EDA
analyze(df, target_col="price")
```

---

## Usage Examples

### How to use
[[Documentation](https://github.com/Akash-Sare03/datasense/blob/main/documentation.md)]

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

This project is licensed under the MIT License. See [LICENSE](LICENSE.txt) for details.

---

## Deployment

This package is available on PyPI. Install it via:

```bash
pip install datasense-eda
```

Or deploy locally:

```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```

---

## Support

If you have any questions or issues, please open an issue [here](https://github.com/Akash-Sare03/datasense/issues).

