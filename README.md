# First Conda Project

A comprehensive data science learning project covering NumPy, Pandas, and Matplotlib through hands-on exercises with real-world datasets including car sales and medical heart disease data.

## Project Overview

This project contains Jupyter notebooks and datasets designed to help learn fundamental data science skills using Python's most popular libraries for data manipulation, analysis, and visualization. From basic array operations to advanced medical data analysis, this project provides a complete learning path for aspiring data scientists.

## What You'll Learn

By working through this project, you'll gain hands-on experience with:

### Core Data Science Stack
- **NumPy**: Array operations, broadcasting, mathematical operations, random number generation
- **Pandas**: DataFrames, data cleaning, handling missing values, data type conversions, aggregations
- **Matplotlib**: Plotting, customization, subplots, statistical visualizations, style themes
- **Scikit-learn**: Classification, regression, model evaluation, hyperparameter tuning, pipelines

### Real-World Skills
- 📊 **Data Preprocessing**: Handling missing data, encoding categorical variables, feature engineering
- 🤖 **Machine Learning Workflows**: Train-test splits, cross-validation, model comparison, hyperparameter optimization
- 📈 **Model Evaluation**: Confusion matrices, ROC curves, precision/recall/F1, regression metrics (MAE, MSE, R²)
- 🔧 **Production-Ready Practices**: Pipelines, model persistence, reproducible workflows with random seeds
- 🐛 **Debugging**: Common errors, version compatibility, troubleshooting deprecated functions

### Key Techniques
- Building end-to-end ML pipelines from raw data to predictions
- Comparing multiple algorithms to find the best model
- Creating reusable evaluation functions for consistent model assessment
- Visualizing model performance with professional plots
- Working with real-world datasets (medical, automotive, housing)

## Prerequisites

### Required Knowledge
- **Python Basics**: Variables, functions, loops, conditionals, lists, dictionaries
- **Basic Math**: Algebra, understanding of mean/median, basic statistics concepts
- **Willingness to Learn**: Curiosity and patience for troubleshooting

### Recommended (but not required)
- Basic understanding of statistics (helpful for ML concepts)
- Familiarity with Jupyter Notebooks
- Command line basics for environment setup

### Software Requirements
- Python 3.10+
- Anaconda or Miniconda installed
- 2-3 GB free disk space
- Text editor or IDE (VS Code, PyCharm, or Jupyter Lab)

## Contents

### Notebooks
- **`Introduction-to-numpy.ipynb`** - Learn NumPy basics including array creation, manipulation, and operations
- **`Introduction_to_pandas.ipynb`** - Explore Pandas fundamentals for data analysis and manipulation
- **`Introduction_to_Matplotlib.ipynb`** - Comprehensive data visualization with Matplotlib, featuring NumPy arrays, car sales analysis, and advanced medical heart disease visualizations with subplots
- **`Introduction to scikitlearn.ipynb`** - Complete machine learning introduction covering classification and regression with multiple algorithms (RandomForest, Ridge, LinearSVC), comprehensive model evaluation metrics (accuracy, precision, recall, F1, ROC/AUC, MAE, MSE, R²), confusion matrix visualization with seaborn, ROC curve plotting, data preprocessing, missing data imputation, model comparison, cross-validation techniques, custom evaluation functions, manual train/validation/test splits (70/15/15), hyperparameter tuning with RandomizedSearchCV, model optimization workflows, scikit-learn Pipelines for end-to-end ML workflows combining preprocessing and modeling, and model persistence with pickle and joblib for real-world datasets including heart disease and California housing
- **`numpy-exercises.ipynb`** - Practice exercises for NumPy concepts and array operations
- **`pandas-exercises.ipynb`** - Additional practice exercises with Pandas
- **`matplotlib-exercises.ipynb`** - Comprehensive Matplotlib exercises covering plotting techniques, customization, styling, and advanced visualization methods including scatter plots, histograms, subplots, and statistical indicators
- **`scikit-learn-exercises.ipynb`** - Hands-on practice exercises for scikit-learn covering end-to-end classification and regression workflows, model comparison, hyperparameter tuning, evaluation metrics, Pipeline building, and model persistence
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
- **`rs_model.pkl`** - Optimized RandomForest model with hyperparameters tuned via RandomizedSearchCV
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
  - Seaborn - Statistical data visualization
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
- Regression analysis with RandomForestRegressor on car sales and housing data
- Train-test data splitting and model evaluation
- Model performance metrics (accuracy, precision, recall, f1-score)
- Confusion matrix analysis and classification reports
- ROC (Receiver Operating Characteristic) curves and AUC scores
- Confusion matrix visualization with seaborn heatmaps
- ConfusionMatrixDisplay for modern matrix visualization
- Hyperparameter tuning and model optimization (n_estimators tuning)
- Data preprocessing with OneHotEncoder and ColumnTransformer
- Handling categorical variables in machine learning
- Missing data imputation with SimpleImputer (constant, mean, and categorical strategies)
- Advanced model comparison with multiple algorithms (Ridge, LinearSVC, RandomForest)
- Working with built-in datasets (California Housing dataset)
- Model persistence with pickle for saving and loading trained models
- Cross-validation and model comparison techniques (single score vs cross-validation)
- Regression evaluation metrics (MAE, MSE, R² score)
- Probability predictions with predict_proba() for classification
- False Positive Rate (FPR) and True Positive Rate (TPR) analysis
- Custom ROC curve plotting with matplotlib
- Creating custom evaluation functions for multi-metric analysis
- Manual train/validation/test splitting (70/15/15 split strategy)
- Hyperparameter optimization with RandomizedSearchCV
- Grid search for multiple parameters (max_depth, max_features, min_samples_split, min_samples_leaf)
- Comparing baseline vs optimized model performance
- Workflow for model improvement and optimization
- Building scikit-learn Pipelines for reproducible ML workflows
- Creating separate Pipeline transformers for different feature types (categorical, numeric, custom)
- Combining preprocessing steps with ColumnTransformer
- End-to-end Pipeline combining data preprocessing and model training
- Saving and loading optimized models for production use (pickle and joblib)

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
   - `scikit-learn-exercises.ipynb` for hands-on machine learning practice
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

#### Feature Mismatch Error in train_test_split
**Problem**: `ValueError: X has N features, but [Model] is expecting M features as input` when calling `.predict()`

**Solution**: Check your `train_test_split()` variable assignment. A common typo is duplicating variable names:
```python
# WRONG - y_train appears twice, X_test is missing:
X_train, y_train, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CORRECT - all four variables properly assigned:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

This error occurs because `X_test` retains data from a previous cell (different dataset with different number of features), causing a mismatch when predicting with a model trained on the current dataset.

#### Classifier vs Regressor Mix-up
**Problem**: `ValueError: Unknown label type: continuous` when using RandomForestClassifier

**Solution**: Use the correct model type for your target variable:
```python
# For continuous target values (prices, temperatures, etc.) - use Regressor
model = RandomForestRegressor()

# For discrete classes (0/1, categories) - use Classifier
clf = RandomForestClassifier()
```

#### Missing Function Parameters in RandomizedSearchCV
**Problem**: `TypeError: RandomizedSearchCV.__init__() missing 2 required positional arguments: 'estimator' and 'param_distributions'`

**Solution**: Pass a base estimator (the model) to RandomizedSearchCV, not another RandomizedSearchCV instance:
```python
# WRONG - trying to use RandomizedSearchCV as an estimator:
clf = RandomizedSearchCV(n_jobs=1)
rs_clf = RandomizedSearchCV(estimator=clf, ...)

# CORRECT - use the actual model as the estimator:
clf = RandomForestClassifier(n_jobs=1)
rs_clf = RandomizedSearchCV(estimator=clf, param_distributions=grid, ...)
```

#### UnboundLocalError in Custom Functions
**Problem**: `UnboundLocalError: cannot access local variable 'accuracy' where it is not associated with a value`

**Solution**: Use the correct function name from sklearn.metrics:
```python
# WRONG - 'accuracy' is not a function:
def evaluate_preds(y_true, y_preds):
    accuracy = accuracy(y_true, y_preds)  # Error!

# CORRECT - use 'accuracy_score':
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_preds(y_true, y_preds):
    accuracy = accuracy_score(y_true, y_preds)  # Correct!
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
```

#### Pipeline Feature Duplication Error
**Problem**: `TypeError: Encoders require their input argument must be uniformly strings or numbers. Got ['float', 'str']`

**Solution**: Don't list the same feature in multiple transformers within a ColumnTransformer:
```python
# WRONG - "Doors" is in both categorical_features and door_feature:
categorical_features = ["Make", "Colour", "Doors"]
door_feature = ["Doors"]

# CORRECT - each feature should only be in one transformer:
categorical_features = ["Make", "Colour"]  # Removed "Doors"
door_feature = ["Doors"]  # Doors has its own transformer

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("door", door_transformer, door_feature),
        ("num", numeric_transformer, numeric_features)
    ]
)
```

#### Dictionary Unpacking for Model Parameters
**Problem**: `InvalidParameterError: The 'penalty' parameter of LogisticRegression must be a str among {...}. Got {'solver': 'liblinear', 'C': ...} instead.`

**Solution**: Use `**` to unpack dictionary parameters when passing best_params_ to a model:
```python
# WRONG - passing dictionary as first positional argument:
clf = LogisticRegression(rs_log_reg.best_params_)

# CORRECT - unpack dictionary with ** operator:
clf = LogisticRegression(**rs_log_reg.best_params_)
```

The `**` operator unpacks `{'solver': 'liblinear', 'C': 3792.69}` into individual keyword arguments: `LogisticRegression(solver='liblinear', C=3792.69)`

#### Deprecated plot_roc_curve Function
**Problem**: `ImportError: cannot import name 'plot_roc_curve' from 'sklearn.metrics'`

**Solution**: Use `RocCurveDisplay` instead (scikit-learn 1.2+ removed `plot_roc_curve`):
```python
# OLD (deprecated in scikit-learn 1.2+):
from sklearn.metrics import plot_roc_curve
plot_roc_curve(model, X_test, y_test)

# NEW (current API):
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(model, X_test, y_test)
```

Similarly, for confusion matrices, use `ConfusionMatrixDisplay.from_estimator()` instead of the deprecated `plot_confusion_matrix()`.

## Contributing

This is a learning project. Feel free to add your own exercises, datasets, or improvements to the notebooks.

## License

This project is for educational purposes.
