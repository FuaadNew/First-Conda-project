# Scikit-Learn Important Functions Reference

A quick reference guide for essential scikit-learn functions used in machine learning workflows.

## Table of Contents
1. [Data Preparation](#data-preparation)
2. [Model Training & Evaluation](#model-training--evaluation)
3. [Preprocessing & Pipelines](#preprocessing--pipelines)
4. [Hyperparameter Tuning & Model Persistence](#hyperparameter-tuning--model-persistence)

---

## Data Preparation

### train_test_split()
**Purpose:** Split data into training and testing sets

**Import:**
```python
from sklearn.model_selection import train_test_split
```

**Basic Usage:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Key Parameters:**
- `test_size`: Proportion of data for testing (e.g., 0.2 = 20%)
- `random_state`: Seed for reproducibility
- `shuffle`: Whether to shuffle data before splitting (default=True)
- `stratify`: Ensures class distribution is maintained (pass `y` for classification)

**Example with stratification:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
```

---

