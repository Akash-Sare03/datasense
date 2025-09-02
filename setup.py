from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="datasense",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "seaborn",
        "matplotlib",
        "ipython"
    ],
    description="An Explainable EDA library for beginners to understand datasets easily.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Akash Sare",
    author_email="akashsare03@gmail.com",
    url="https://github.com/Akash-Sare03/datasense", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    keywords="eda, data-science, exploratory-data-analysis, machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/Akash-Sare03/datasense/issues",
        "Source": "https://github.com/Akash-Sare03/datasense",
    },
)
