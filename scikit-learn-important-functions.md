# Scikit-Learn Important Functions Reference

A quick reference guide for essential scikit-learn functions used in machine learning workflows.

## Table of Contents
1. [Data Preparation](#data-preparation)
2. [Model Training & Evaluation](#model-training--evaluation)
3. [Preprocessing & Pipelines](#preprocessing--pipelines)
4. [Clustering Algorithms](#clustering-algorithms)
5. [Hyperparameter Tuning & Model Persistence](#hyperparameter-tuning--model-persistence)

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

#### StandardScaler()
**Purpose:** Standardize features by removing mean and scaling to unit variance

**Import:**
```python
from sklearn.preprocessing import StandardScaler
```

**Basic Usage:**
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Key Points:**
- Transforms data to have mean=0 and std=1
- Essential for algorithms sensitive to feature scales (SVM, KNN, Logistic Regression)
- Formula: z = (x - mean) / std
- **IMPORTANT:** Fit on training data only, then transform both train and test

**Example:**
```python
# Before scaling: [10, 20, 30, 40, 50]
# After scaling: [-1.41, -0.71, 0, 0.71, 1.41]
```

#### MinMaxScaler()
**Purpose:** Scale features to a given range (default 0-1)

**Import:**
```python
from sklearn.preprocessing import MinMaxScaler
```

**Basic Usage:**
```python
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Key Points:**
- Scales features to a fixed range (default [0, 1])
- Formula: X_scaled = (X - X_min) / (X_max - X_min)
- Useful when you need bounded values
- More affected by outliers than StandardScaler

**Example with custom range:**
```python
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)
```

**StandardScaler vs MinMaxScaler:**
- Use StandardScaler: For most ML algorithms, when data has outliers
- Use MinMaxScaler: When you need specific range (e.g., neural networks), when data doesn't have outliers

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

## Clustering Algorithms

### KMeans()
**Purpose:** Partition data into K distinct clusters based on feature similarity

**Import:**
```python
from sklearn.cluster import KMeans
```

**Basic Usage:**
```python
# Create KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

# Get cluster centers
centers = kmeans.cluster_centers_

# Predict cluster for new data
new_labels = kmeans.predict(X_new)
```

**Key Parameters:**
- `n_clusters`: Number of clusters to form (default=8)
- `random_state`: Seed for reproducibility
- `n_init`: Number of times algorithm runs with different centroid seeds (default=10)
- `max_iter`: Maximum iterations for convergence (default=300)

**Finding Optimal K (Elbow Method):**
```python
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Plot to find "elbow"
import matplotlib.pyplot as plt
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

**Use Cases:**
- Customer segmentation
- Image compression
- Document clustering
- Market segmentation

### DBSCAN()
**Purpose:** Density-based clustering that finds core samples and groups them together

**Import:**
```python
from sklearn.cluster import DBSCAN
```

**Basic Usage:**
```python
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# -1 indicates noise/outliers
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
```

**Key Parameters:**
- `eps`: Maximum distance between two samples to be considered neighbors
- `min_samples`: Minimum samples in neighborhood for a core point
- `metric`: Distance metric (default='euclidean')

**Advantages over KMeans:**
- Doesn't require specifying number of clusters
- Can find arbitrarily shaped clusters
- Identifies outliers as noise points
- Robust to clusters of different sizes

**Use Cases:**
- Anomaly detection
- Geographic data clustering
- Finding clusters of arbitrary shape
- When number of clusters is unknown

**Example: Comparing KMeans and DBSCAN:**
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons

# Generate non-linear separable data
X, y = make_moons(n_samples=200, noise=0.05, random_state=42)

# KMeans struggles with non-spherical clusters
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# DBSCAN handles non-spherical clusters well
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)
```

---

## Hyperparameter Tuning & Model Persistence

### Hyperparameter Tuning

#### RandomizedSearchCV()
**Purpose:** Randomly search through hyperparameter space to find best model

**Import:**
```python
from sklearn.model_selection import RandomizedSearchCV
```

**Basic Usage:**
```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Define hyperparameter grid
param_grid = {
    'n_estimators': [10, 100, 200, 500, 1000],
    'max_depth': [None, 5, 10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

# Create base model
clf = RandomForestClassifier(random_state=42)

# Create RandomizedSearchCV
rs_clf = RandomizedSearchCV(
    estimator=clf,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit
rs_clf.fit(X_train, y_train)

# Get best parameters and score
print(rs_clf.best_params_)
print(rs_clf.best_score_)
```

**Key Parameters:**
- `estimator`: The model to tune
- `param_distributions`: Dictionary of hyperparameters to search
- `n_iter`: Number of random combinations to try
- `cv`: Number of cross-validation folds
- `scoring`: Metric to optimize (default='accuracy')
- `n_jobs`: Number of parallel jobs (-1 = use all cores)

**Using Best Parameters:**
```python
# IMPORTANT: Use ** to unpack the dictionary
best_model = RandomForestClassifier(**rs_clf.best_params_)
best_model.fit(X_train, y_train)
```

#### GridSearchCV()
**Purpose:** Exhaustively search through all hyperparameter combinations

**Import:**
```python
from sklearn.model_selection import GridSearchCV
```

**Basic Usage:**
```python
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```

**Difference from RandomizedSearchCV:**
- GridSearchCV: Tests ALL combinations (exhaustive, slower)
- RandomizedSearchCV: Tests RANDOM subset (faster, good for large grids)

### Model Persistence

#### joblib.dump() & joblib.load()
**Purpose:** Save and load trained models to disk

**Import:**
```python
import joblib
```

**Saving a Model:**
```python
# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save to disk
joblib.dump(model, 'my_model.pkl')
```

**Loading a Model:**
```python
# Load from disk
loaded_model = joblib.load('my_model.pkl')

# Use it
predictions = loaded_model.predict(X_test)
```

**Saving a Pipeline:**
```python
# Pipelines can be saved the same way
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)

# Save entire pipeline
joblib.dump(pipeline, 'full_pipeline.pkl')

# Load and use
loaded_pipeline = joblib.load('full_pipeline.pkl')
predictions = loaded_pipeline.predict(X_test)
```

**Best Practices:**
- Use `.pkl` file extension for model files
- `joblib` is preferred over `pickle` for large numpy arrays
- Save the entire pipeline (including preprocessing) for consistency
- Keep track of scikit-learn version used for training

#### Alternative: pickle
**Import:**
```python
import pickle
```

**Basic Usage:**
```python
# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
```

**Note:** `joblib` is generally faster for scikit-learn models with large numpy arrays.

---

## Common Pitfalls & Pro Tips

### 1. Dictionary Unpacking for Best Parameters
```python
# WRONG - passes dict as single argument
model = LogisticRegression(rs.best_params_)

# CORRECT - unpacks dict as keyword arguments
model = LogisticRegression(**rs.best_params_)
```

### 2. Classifier vs Regressor
```python
# For continuous targets (prices, temperatures, etc.)
model = RandomForestRegressor()

# For discrete classes (yes/no, categories, etc.)
model = RandomForestClassifier()
```

### 3. Pipeline Step Format
```python
# WRONG - not a tuple
steps = ["preprocessor", model]

# CORRECT - list of tuples
steps = [("preprocessor", preprocessor), ("model", model)]
```

### 4. Cross-validation Scoring Parameter
```python
# WRONG
cross_val_score(model, X, y, scoring='f1_score')

# CORRECT
cross_val_score(model, X, y, scoring='f1')
```

### 5. ROC Curve (Updated API)
```python
# OLD (deprecated)
from sklearn.metrics import plot_roc_curve
plot_roc_curve(model, X_test, y_test)

# NEW (scikit-learn 1.2+)
from sklearn.metrics import RocCurveDisplay
RocCurveDisplay.from_estimator(model, X_test, y_test)
```

---

## Quick Workflow Template

```python
# 1. Import libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 2. Load and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define model and hyperparameters
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20]
}

# 4. Tune hyperparameters
rs = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    n_iter=5,
    cv=5
)
rs.fit(X_train, y_train)

# 5. Train final model
final_model = RandomForestClassifier(**rs.best_params_)
final_model.fit(X_train, y_train)

# 6. Evaluate
y_pred = final_model.predict(X_test)
print(classification_report(y_test, y_pred))

# 7. Save model
joblib.dump(final_model, 'final_model.pkl')
```

---

**Created:** November 2025  
**For:** firstCondaProject scikit-learn learning path

