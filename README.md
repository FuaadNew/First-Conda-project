# First Conda Project

A comprehensive data science learning project covering NumPy, Pandas, and Matplotlib through hands-on exercises with real-world datasets including car sales and medical heart disease data.

## Project Overview

This project contains Jupyter notebooks and datasets designed to help learn fundamental data science skills using Python's most popular libraries for data manipulation, analysis, and visualization. From basic array operations to advanced medical data analysis, this project provides a complete learning path for aspiring data scientists.

## Contents

### Notebooks
- **`Introduction-to-numpy.ipynb`** - Learn NumPy basics including array creation, manipulation, and operations
- **`Introduction_to_pandas.ipynb`** - Explore Pandas fundamentals for data analysis and manipulation
- **`Introduction_to_Matplotlib.ipynb`** - Comprehensive data visualization with Matplotlib, featuring NumPy arrays, car sales analysis, and advanced medical heart disease visualizations with subplots
- **`numpy-exercises.ipynb`** - Practice exercises for NumPy concepts and array operations
- **`pandas-exercises.ipynb`** - Additional practice exercises with Pandas
- **`Untitled.ipynb`** - Scratch notebook for experimentation

### Datasets
- **`car-sales.csv`** - Complete car sales dataset with make, color, odometer, doors, and price information
- **`car-sales-missing-data.csv`** - Car sales dataset with intentionally missing values for practicing data cleaning techniques
- **`heart-disease.csv`** - Medical dataset containing cardiovascular health indicators (age, cholesterol, heart rate, etc.) for advanced data analysis, visualization, and machine learning practice

### Assets
- **`leetcode200.png`** - Reference image

## Environment Setup

This project uses Conda for environment management. The environment is defined in `environment.yaml`.

### Prerequisites
- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Installation

1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd firstCondaProject
   ```

3. Create the conda environment:
   ```bash
   conda env create -f environment.yaml
   ```

4. Activate the environment:
   ```bash
   conda activate firstCondaProject
   ```

5. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Dependencies

- **Python 3.10**
- **Core Libraries:**
  - NumPy - Numerical computing
  - Pandas - Data manipulation and analysis
  - Matplotlib - Data visualization
  - Scikit-learn - Machine learning
- **Additional:**
  - Requests - HTTP library
  - Pip - Package management

## Learning Objectives

### NumPy
- Understanding NumPy arrays and their advantages
- Array creation methods (zeros, ones, ranges)
- Array manipulation and operations
- Working with multi-dimensional arrays

### Pandas
- Loading and exploring datasets
- Data cleaning and preprocessing
- Data type conversion and string manipulation
- Handling missing data
- Basic data analysis techniques

### Matplotlib
- Creating basic plots and visualizations
- Working with figures and axes
- Plotting with NumPy arrays
- Advanced subplot layouts for multi-panel visualizations
- Customizing plot appearance (titles, labels, sizing, legends)
- Adding reference lines and statistical indicators
- Saving plots as image files
- Visualizing real-world data from CSV files
- Medical data visualization and correlation analysis

## Getting Started

1. Start with `Introduction-to-numpy.ipynb` to learn array fundamentals
2. Progress to `Introduction_to_pandas.ipynb` for data manipulation
3. Continue with `Introduction_to_Matplotlib.ipynb` to learn data visualization
4. Practice with `numpy-exercises.ipynb` and `pandas-exercises.ipynb` for reinforcement
5. Experiment with the car sales and heart disease datasets to apply your skills

## Dataset Information

The car sales datasets contain the following columns:
- **Make**: Car manufacturer (Toyota, Honda, BMW, Nissan)
- **Colour**: Car color
- **Odometer**: Mileage in kilometers
- **Doors**: Number of doors
- **Price**: Sale price in USD

The missing data version is specifically designed for practicing data cleaning techniques including handling null values and inconsistent formatting.

## Tips for Learning

- Run each cell in the notebooks sequentially
- Experiment with the code by modifying parameters
- Try to predict outputs before running cells
- Use the datasets to practice new concepts
- Don't hesitate to add your own cells for experimentation

## Troubleshooting

### Common Issues with Python/Pandas Version Differences

When following tutorials from 2020-2022, you may encounter some compatibility issues with newer Python and Pandas versions:

#### String Regex Patterns
**Problem**: `SyntaxWarning: invalid escape sequence` when using regex patterns like `'[\$\,\.]'`

**Solution**: Use raw strings with the `r` prefix:
```python
# Instead of: df["Price"].str.replace('[\$\,\.]', '')
# Use:
df["Price"] = df["Price"].str.replace(r'[\$\,\.]', '', regex=True)
```

#### Data Type Conversion
**Problem**: `TypeError: can only concatenate str (not "float") to str` when performing mathematical operations

**Solution**: Convert strings to numeric after cleaning:
```python
# Clean the data first
df["Price"] = df["Price"].str.replace(r'[\$\,\.]', '', regex=True)
# Then convert to numeric
df["Price"] = pd.to_numeric(df["Price"])
```

#### Function Parameter Changes
**Problem**: Some pandas functions now require explicit parameters that were optional before

**Solution**: Always specify the `regex=True` parameter when using regex patterns in `str.replace()`

## Contributing

This is a learning project. Feel free to add your own exercises, datasets, or improvements to the notebooks.

## License

This project is for educational purposes.
