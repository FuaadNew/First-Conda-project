# First Conda Project

A comprehensive data science learning project covering NumPy, Pandas, and Matplotlib through hands-on exercises with real-world datasets including car sales and medical heart disease data.

## Project Overview

This project contains Jupyter notebooks and datasets designed to help learn fundamental data science skills using Python's most popular libraries for data manipulation, analysis, and visualization. From basic array operations to advanced medical data analysis, this project provides a complete learning path for aspiring data scientists.

## Contents

### Notebooks
- **`Introduction-to-numpy.ipynb`** - Learn NumPy basics including array creation, manipulation, and operations
- **`Introduction_to_pandas.ipynb`** - Explore Pandas fundamentals for data analysis and manipulation
- **`Introduction_to_Matplotlib.ipynb`** - Comprehensive data visualization with Matplotlib, featuring NumPy arrays, car sales analysis, and advanced medical heart disease visualizations with subplots
- **`Introduction to scikitlearn.ipynb`** - Complete machine learning introduction covering classification and regression with multiple algorithms (RandomForest, Ridge, LinearSVC), model evaluation metrics, data preprocessing, missing data imputation, model comparison, cross-validation, and model persistence with real-world datasets including heart disease and California housing
- **`numpy-exercises.ipynb`** - Practice exercises for NumPy concepts and array operations
- **`pandas-exercises.ipynb`** - Additional practice exercises with Pandas
- **`matplotlib-exercises.ipynb`** - Comprehensive Matplotlib exercises covering plotting techniques, customization, styling, and advanced visualization methods including scatter plots, histograms, subplots, and statistical indicators
- **`Untitled.ipynb`** - Scratch notebook for experimentation

### Datasets
- **`car-sales.csv`** - Complete car sales dataset with make, color, odometer, doors, and price information
- **`car-sales-missing-data.csv`** - Car sales dataset with intentionally missing values for practicing data cleaning techniques
- **`car-sales-extended-missing-data.csv`** - Extended car sales dataset with additional missing data scenarios for advanced data cleaning practice
- **`heart-disease.csv`** - Medical dataset containing cardiovascular health indicators (age, cholesterol, heart rate, etc.) for advanced data analysis, visualization, and machine learning practice

### Generated Assets & Images
- **`helloworldplot.png`** - Sample plot generated from matplotlib exercises
- **`test_scatter.png`** - Scatter plot example from matplotlib practice
- **`leetcode200.png`** - Reference image
- **`random_forest_model_1.pkl`** - Trained RandomForest model saved using pickle for model persistence
- **`images/`** - Directory containing additional sample plots and reference images
  - **`sample-plot.png`** - Example visualization output

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
- Creating basic plots and visualizations (line, scatter, bar, histogram)
- Working with figures and axes (both pyplot and object-oriented approaches)
- Plotting with NumPy arrays and Pandas DataFrames
- Advanced subplot layouts for multi-panel visualizations
- Customizing plot appearance (titles, labels, sizing, legends, colors)
- Adding reference lines and statistical indicators (mean lines, trend lines)
- Plot styling and themes (seaborn integration, custom color schemes)
- Saving plots as image files in various formats
- Visualizing real-world data from CSV files
- Medical data visualization and correlation analysis
- Interactive plotting techniques and best practices

### Scikit-learn (Machine Learning)
- Binary classification with RandomForestClassifier on heart disease data
- Regression analysis with RandomForestRegressor on car sales data
- Train-test data splitting and model evaluation
- Model performance metrics (accuracy, precision, recall, f1-score)
- Confusion matrix analysis and classification reports
- Hyperparameter tuning and model optimization
- Data preprocessing with OneHotEncoder and ColumnTransformer
- Handling categorical variables in machine learning
- Missing data imputation with SimpleImputer (constant, mean, and categorical strategies)
- Advanced model comparison with multiple algorithms (Ridge, LinearSVC, RandomForest)
- Working with built-in datasets (California Housing dataset)
- Model persistence with pickle for saving and loading trained models
- Cross-validation and model comparison techniques
- Mean absolute error (MAE) for regression evaluation

## Getting Started

### Recommended Learning Path

1. **Start with `Introduction-to-numpy.ipynb`** to learn array fundamentals
2. **Progress to `Introduction_to_pandas.ipynb`** for data manipulation
3. **Continue with `Introduction_to_Matplotlib.ipynb`** to learn basic data visualization
4. **Advance to `Introduction to scikitlearn.ipynb`** for machine learning fundamentals
5. **Practice with the exercise notebooks:**
   - `numpy-exercises.ipynb` for NumPy reinforcement
   - `pandas-exercises.ipynb` for Pandas practice
   - `matplotlib-exercises.ipynb` for advanced plotting techniques
6. **Apply your skills** with the car sales and heart disease datasets
7. **Experiment freely** using `Untitled.ipynb` as your sandbox

### Quick Start
```bash
# Clone and setup
cd firstCondaProject
conda env create -f environment.yaml
conda activate firstCondaProject
jupyter notebook
```

## Dataset Information

### Car Sales Dataset
The car sales datasets contain the following columns:
- **Make**: Car manufacturer (Toyota, Honda, BMW, Nissan)
- **Colour**: Car color
- **Odometer**: Mileage in kilometers
- **Doors**: Number of doors
- **Price**: Sale price in USD

Three versions are available:
- **Complete dataset** (`car-sales.csv`) - Clean data for baseline analysis
- **Missing data** (`car-sales-missing-data.csv`) - Designed for practicing basic data cleaning techniques including handling null values and inconsistent formatting
- **Extended missing data** (`car-sales-extended-missing-data.csv`) - Additional missing data scenarios for advanced cleaning practice

### Heart Disease Dataset
The heart disease dataset contains medical indicators for cardiovascular health prediction:
- **target**: Binary classification target (0 = no heart disease, 1 = heart disease)
- Various health metrics including age, cholesterol levels, heart rate, and other cardiovascular indicators
- Used for binary classification machine learning tasks and medical data analysis

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

#### Scikit-learn Class Instantiation
**Problem**: `AttributeError: 'DataFrame' object has no attribute '_validate_params'` when calling `.fit()`

**Solution**: Make sure to instantiate classifier/regressor classes with parentheses:
```python
# Instead of: clf = RandomForestClassifier
# Use:
clf = RandomForestClassifier()
```

## Contributing

This is a learning project. Feel free to add your own exercises, datasets, or improvements to the notebooks.

## License

This project is for educational purposes.
