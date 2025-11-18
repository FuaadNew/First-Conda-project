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

## Model Training & Evaluation

### Classification Models

#### RandomForestClassifier()
**Purpose:** Ensemble decision tree classifier

**Import:**
```python
from sklearn.ensemble import RandomForestClassifier
```

**Basic Usage:**
```python
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

**Key Parameters:**
- `n_estimators`: Number of trees in the forest (default=100)
- `max_depth`: Maximum depth of trees (default=None)
- `random_state`: Seed for reproducibility
- `n_jobs`: Number of parallel jobs (-1 uses all processors)

#### LogisticRegression()
**Purpose:** Linear classification model

**Import:**
```python
from sklearn.linear_model import LogisticRegression
```

**Basic Usage:**
```python
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
```

**Key Parameters:**
- `C`: Inverse regularization strength (smaller = stronger regularization)
- `solver`: Algorithm to use ('liblinear', 'lbfgs', etc.)
- `max_iter`: Maximum iterations for convergence

### Regression Models

#### RandomForestRegressor()
**Purpose:** Ensemble decision tree regressor for continuous targets

**Import:**
```python
from sklearn.ensemble import RandomForestRegressor
```

**Basic Usage:**
```python
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)
```

#### Ridge()
**Purpose:** Linear regression with L2 regularization

**Import:**
```python
from sklearn.linear_model import Ridge
```

**Basic Usage:**
```python
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
```

### Evaluation Metrics

#### Classification Metrics

**Import:**
```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
```

**accuracy_score()** - Percentage of correct predictions
```python
accuracy = accuracy_score(y_true, y_pred)
```

**precision_score()** - True positives / (True positives + False positives)
```python
precision = precision_score(y_true, y_pred)
```

**recall_score()** - True positives / (True positives + False negatives)
```python
recall = recall_score(y_true, y_pred)
```

**f1_score()** - Harmonic mean of precision and recall
```python
f1 = f1_score(y_true, y_pred)
```

**confusion_matrix()** - Matrix showing prediction vs actual
```python
cm = confusion_matrix(y_true, y_pred)
```

**classification_report()** - Comprehensive text report
```python
report = classification_report(y_true, y_pred)
print(report)
```

**ROC Curve Display:**
```python
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show()
```

#### Regression Metrics

**Import:**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

**mean_absolute_error()** - Average absolute difference
```python
mae = mean_absolute_error(y_true, y_pred)
```

**mean_squared_error()** - Average squared difference
```python
mse = mean_squared_error(y_true, y_pred)
```

**r2_score()** - R² coefficient (proportion of variance explained)
```python
r2 = r2_score(y_true, y_pred)
```

### Cross-Validation

**cross_val_score()** - Evaluate model using k-fold cross-validation

**Import:**
```python
from sklearn.model_selection import cross_val_score
```

**Basic Usage:**
```python
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
mean_score = scores.mean()
```

**Key Parameters:**
- `cv`: Number of folds (default=5)
- `scoring`: Metric to use ('accuracy', 'f1', 'precision', 'recall', 'r2', etc.)

---

## Preprocessing & Pipelines

### Data Preprocessing

#### SimpleImputer()
**Purpose:** Fill in missing values in datasets

**Import:**
```python
from sklearn.impute import SimpleImputer
```

**Basic Usage:**
```python
# For numerical features - fill with median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
```

**Key Parameters & Strategies:**
- `strategy='mean'`: Fill with column mean
- `strategy='median'`: Fill with column median (robust to outliers)
- `strategy='most_frequent'`: Fill with most common value
- `strategy='constant'`, `fill_value='missing'`: Fill with custom value

**Example for categorical data:**
```python
cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
```

#### OneHotEncoder()
**Purpose:** Convert categorical variables to binary columns

**Import:**
```python
from sklearn.preprocessing import OneHotEncoder
```

**Basic Usage:**
```python
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_categorical)
```

**Key Parameters:**
- `handle_unknown='ignore'`: Ignore unknown categories during transform
- `sparse_output=False`: Return dense array instead of sparse matrix

**Example:**
```python
# If you have categories: ['red', 'blue', 'green']
# OneHotEncoder creates: [1,0,0], [0,1,0], [0,0,1]
```

### Pipelines

#### Pipeline()
**Purpose:** Chain multiple preprocessing steps and model together

**Import:**
```python
from sklearn.pipeline import Pipeline
```

**Basic Usage:**
```python
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**Key Benefits:**
- Prevents data leakage (fit only on training data)
- Simplifies workflow (one object for preprocessing + model)
- Makes code cleaner and more reproducible

**Step Format:** Each step is a tuple: `('name', transformer_or_model)`

#### ColumnTransformer()
**Purpose:** Apply different preprocessing to different columns

**Import:**
```python
from sklearn.compose import ColumnTransformer
```

**Basic Usage:**
```python
# Define transformers for different column types
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine them
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Use in a full pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

model.fit(X_train, y_train)
```

**Transformer Format:** Each transformer is a tuple:
```python
('name', transformer, columns)
```

**Complete Example:**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Define feature lists
numeric_features = ['Age', 'Income', 'CreditScore']
categorical_features = ['Gender', 'City', 'ProductCategory']

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
full_pipeline.fit(X_train, y_train)

# Predict
predictions = full_pipeline.predict(X_test)

# Score
score = full_pipeline.score(X_test, y_test)
```

---

