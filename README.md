# First Conda Project

A beginner-friendly data science project focusing on learning NumPy and Pandas through hands-on exercises with car sales data.

## Project Overview

This project contains Jupyter notebooks and datasets designed to help learn fundamental data science skills using Python's most popular libraries for data manipulation and analysis.

## Contents

### Notebooks
- **`Introduction-to-numpy.ipynb`** - Learn NumPy basics including array creation, manipulation, and operations
- **`Introduction_to_pandas.ipynb`** - Explore Pandas fundamentals for data analysis and manipulation
- **`pandas-exercises.ipynb`** - Additional practice exercises with Pandas
- **`Untitled.ipynb`** - Scratch notebook for experimentation

### Datasets
- **`car-sales.csv`** - Complete car sales dataset with make, color, odometer, doors, and price information
- **`car-sales-missing-data.csv`** - Car sales dataset with intentionally missing values for practicing data cleaning techniques

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

## Getting Started

1. Start with `Introduction-to-numpy.ipynb` to learn array fundamentals
2. Progress to `Introduction_to_pandas.ipynb` for data manipulation
3. Practice with `pandas-exercises.ipynb` for reinforcement
4. Experiment with the car sales datasets to apply your skills

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

## Contributing

This is a learning project. Feel free to add your own exercises, datasets, or improvements to the notebooks.

## License

This project is for educational purposes.
