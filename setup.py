from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Extract version from datasense/__init__.py
def get_version():
    about = {}
    with open(os.path.join("datasense", "__init__.py")) as f:
        exec(f.read(), about)
    return about["__version__"]

setup(
    name="datasense",
    version=get_version(),  # synced with __init__.py
    author="Akash Sare",
    author_email="akashsare03@gmail.com",
    description="Datasense: An Explainable EDA library for automated exploratory data analysis, outlier detection, feature importance, and visualization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Akash-Sare03/datasense",
    project_urls={
        "Bug Tracker": "https://github.com/Akash-Sare03/datasense/issues",
        "Source Code": "https://github.com/Akash-Sare03/datasense",
        "Documentation": "https://github.com/Akash-Sare03/datasense#readme",
    },
    license="MIT",
    keywords="eda, data-analysis, data-science, machine-learning, visualization, feature-importance, outlier-detection, time-series",
    packages=find_packages(exclude=("tests", "notebooks", "examples")),
    install_requires=[
        "pandas>=1.2.0",
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "scikit-learn>=0.24.0",
        "statsmodels>=0.12.0",
        "seaborn>=0.11.0",
        "matplotlib>=3.3.0",
        "ipython>=7.0.0",
        "tabulate"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
