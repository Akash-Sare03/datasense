# 📖 DataSense Documentation

A Python library for explainable and automated Exploratory Data Analysis (EDA).
This guide provides detailed usage instructions with examples.

📌 Table of Contents

Installation

Quick Start

Core Functions

1. Dataset Summary

2. Statistical Analysis

3. Missing Values Handling

4. Outlier Detection and Handling

5. Feature Importance

6. Automated Recommendations

7. Comprehensive Analysis

8. Time Series Analysis

9. Visualization Functions

Complete Example Workflow

Advanced Usage

Best Practices

Troubleshooting

# 🚀 Installation
pip install datasense-eda

# ⚡ Quick Start

```python
import datasense as ds
import pandas as pd

# Load dataset
df = pd.read_csv("your_dataset.csv")

# Comprehensive EDA Report
ds.analyze(df)

# Dataset summary
from ds.summary import summarize_dataset
summary = summarize_dataset(df)
print(summary)
```
# 🔑 Core Functions
## 1. Dataset Summary
```python
summarize_dataset(data, include_sample=True, sample_rows=5)


#Generates a detailed Markdown summary report of a dataset.

#Parameters

data: DataFrame, list, dict, or CSV path

include_sample: Show sample rows (default: True)

sample_rows: Number of rows to display (default: 5)

#Example

import datasense as ds
import pandas as pd

data = {
    'age': [25, 30, 35, 40, 45, None, 55],
    'salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000],
    'department': ['IT', 'HR', 'IT', 'Finance', 'HR', 'IT', 'Finance']
}
df = pd.DataFrame(data)

summary = ds.summarize_dataset(df)
print(summary)
```

## 2. Statistical Analysis
```python
calculate_statistics(data, explain=True, columns=None, narrative=True)


#Calculates descriptive statistics for numeric columns.

#Parameters

explain: Include human-readable explanations (default: True)

columns: Specific column(s)

narrative: Add deeper narrative insights (default: True)

#Example

stats = ds.calculate_statistics(df, columns=['age', 'salary'])
print(stats)

age_stats = ds.calculate_statistics(df, columns='age')
```

## 3. Missing Values Handling
```python
find_missing_values(data)
handle_missing_values(data, method="drop", value=None)


#Strategies

"drop" → remove rows

"mean", "median", "mode" → fill with statistics

"constant" → fill with custom value

#Examples

missing_report = ds.find_missing_values(df)
print(missing_report)

df_clean, report = ds.handle_missing_values(df, method="mean")
```

## 4. Outlier Detection and Handling
```python
detect_outliers(data, method="zscore", threshold=3.0, visualize=True)
remove_outliers(data, method="iqr", strategy="remove")


#Strategies

"remove" → delete outliers

"cap" → cap values

"nan" → replace with NaN

#Examples

outlier_report = ds.detect_outliers(df, method="zscore", threshold=2.5)

df_no_outliers, report = ds.remove_outliers(df, strategy="remove")
```

## 5. Feature Importance
```python
feature_importance_calculate(data, target_col, top_n=10, show_bottom=False)


#Example

importance_df, report, target_type = ds.feature_importance_calculate(
    df, target_col='salary', top_n=5
)
```

## 6. Automated Recommendations
```python
recommendations = ds.generate_recommendations(df)
print(recommendations)
```

## 7. Comprehensive Analysis
```python
ds.analyze(df, target_col=None, outlier_method="zscore")


#Example

ds.analyze(df)
ds.analyze(df, target_col=['age', 'salary'])
ds.analyze(df, outlier_method="iqr")
```

## 8. Time Series Analysis
```python
ds.analyze_timeseries(df, date_col, target_col, freq="D", window=7)
```

## 9. Visualization Functions
```python
plots = ds.visualize(df, cols=None, save_plots=False)


#individual Plot Examples

# Missing values plot
fig, md = ds.plot_missing_values(df)

# Histogram
fig, md = ds.plot_histogram(df, 'age')

# Boxplot
fig, md = ds.plot_boxplot(df, 'salary')

# Count plot
fig, md = ds.plot_countplot(df, 'department')

# Correlation matrix
fig, md = ds.plot_correlation_matrix(df)

# Scatter plot
fig, md = ds.plot_scatterplot(df, x='age', y='salary')

# Pair plot
fig, md = ds.plot_pairplot(df, columns=['age', 'salary'])
```

# 🛠 Complete Example Workflow
#### Let’s walk through a full EDA workflow using DataSense on a sample dataset.
```python
import datasense as ds
import pandas as pd
import numpy as np

# Create sample dataset
np.random.seed(42)
data = {
    'age': np.random.normal(35, 10, 100),
    'income': np.random.normal(50000, 15000, 100),
    'experience': np.random.normal(8, 5, 100),
    'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 100),
    'performance_score': np.random.uniform(0, 100, 100)
}
df = pd.DataFrame(data)

# Add some missing values and outliers
df.loc[10:15, 'age'] = np.nan
df.loc[20:25, 'income'] = np.nan
df.loc[5, 'income'] = 200000  # Outlier

print("=== COMPREHENSIVE EDA WORKFLOW ===")

# 1. Dataset summary
print("1. Dataset Summary:")
summary = ds.summarize_dataset(df)
print(summary)

# 2. Missing values analysis
print("\n2. Missing Values Analysis:")
missing_report = ds.find_missing_values(df)
print(missing_report)

# 3. Handle missing values
print("\n3. Handling Missing Values:")
df_clean, handle_report = ds.handle_missing_values(df, method="mean")
print(handle_report)

# 4. Outlier detection
print("\n4. Outlier Detection:")
outlier_report = ds.detect_outliers(df_clean, method="zscore")
print(outlier_report)

# 5. Statistical analysis
print("\n5. Statistical Analysis:")
stats = ds.calculate_statistics(df_clean, columns=['age', 'income'])
print(stats)

# 6. Feature importance
print("\n6. Feature Importance:")
importance_df, importance_report, _ = ds.feature_importance_calculate(
    df_clean, target_col='performance_score', top_n=3
)
print(importance_report)

# 7. Recommendations
print("\n7. Data Recommendations:")
recommendations = ds.generate_recommendations(df_clean)
print(recommendations)

# 8. Visualizations
print("\n8. Generating Visualizations:")
plots = ds.visualize(df_clean, cols=['age', 'income', 'department'])

# 9. Comprehensive analysis
print("\n9. Comprehensive Analysis Report:")
ds.analyze(df_clean)
```

# ✅ Best Practices

-> Always check for missing values

-> Handle outliers based on domain knowledge

-> Use sampling for large datasets

-> Save reports/visualizations for documentation

# 🐞 Troubleshooting

Common Issues

Empty dataset → check if DataFrame loaded

Column not found → verify column names

Memory errors → sample large datasets

Visualization issues → check matplotlib / seaborn install

# Debug Example
```python
try:
    result = ds.analyze(df)
except Exception as e:
    print(f"Error: {e}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
```