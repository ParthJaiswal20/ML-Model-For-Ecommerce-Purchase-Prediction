Decision Tree-based ML model for predicting e-commerce purchase behavior.

# ShopSmart E-commerce Purchase Prediction

## Overview

This project builds a **Machine Learning classification model** to predict whether an e-commerce visitor will complete a purchase based on their browsing and session behavior.

The project uses a **Decision Tree Classifier** with preprocessing, class balancing, hyperparameter tuning, and pruning through tree-complexity control. Since the dataset is imbalanced, **F1-score** is used as the primary evaluation metric.

---

## Objective

The main objectives of this project are:

* Perform exploratory analysis of e-commerce session data.
* Preprocess numerical and categorical features.
* Build a Decision Tree classification model.
* Handle class imbalance using `class_weight="balanced"`.
* Control tree complexity using `max_depth` and `min_samples_leaf`.
* Use GridSearchCV to find suitable hyperparameters.
* Evaluate the final model using F1-score, precision, recall, confusion matrix, and accuracy.

---

## Dataset

The dataset contains **12,330 individual e-commerce user sessions** collected over one year.

### Target Variable

* `Revenue` — indicates whether the visitor completed a purchase.

  * `0` → No purchase
  * `1` → Purchase

### Features

The dataset contains session-level behavioral and categorical features including:

* Administrative
* Administrative_Duration
* Informational
* Informational_Duration
* ProductRelated
* ProductRelated_Duration
* BounceRates
* ExitRates
* PageValues
* SpecialDay
* Month
* OperatingSystems
* Browser
* Region
* TrafficType
* VisitorType
* Weekend

---

## Machine Learning Workflow

The project follows this workflow:

```text
Dataset
   ↓
Feature & Target Separation
   ↓
Identify Numerical & Categorical Features
   ↓
Train-Test Split (80:20)
   ↓
Stratified Sampling
   ↓
Feature Preprocessing
   ├── StandardScaler
   └── OneHotEncoder
   ↓
Decision Tree Classifier
   ↓
Class Balancing
   ↓
GridSearchCV
   ↓
Hyperparameter Tuning / Tree Complexity Control
   ↓
Final Prediction
   ↓
Model Evaluation
```

---

## Data Preprocessing

### Train-Test Split

The dataset was divided into:

* **80% training data**
* **20% testing data**

A `random_state` of `32` was used, along with `stratify=y` to maintain the class distribution between training and testing sets.

### Numerical Features

Numerical features were standardized using:

```python
StandardScaler()
```

### Categorical Features

Categorical features were converted into numerical representations using:

```python
OneHotEncoder(handle_unknown="ignore")
```

Both preprocessing steps were implemented using a `ColumnTransformer`.

---

## Model

### Decision Tree Classifier

The main algorithm used in this project is:

```python
DecisionTreeClassifier()
```

The initial model configuration included:

```python
max_depth=6
min_samples_leaf=30
class_weight="balanced"
random_state=42
```

### Handling Class Imbalance

The dataset is imbalanced, so:

```python
class_weight="balanced"
```

was used to give greater consideration to the minority class during model training.

---

## Hyperparameter Tuning

`GridSearchCV` was used with **5-fold cross-validation** and **F1-score** as the scoring metric.

The following parameters were tested:

```python
param_grid = {
    "model__max_depth": [4, 6, 8],
    "model__min_samples_leaf": [20, 30, 50]
}
```

This also provides **tree-complexity control/pruning** by limiting tree depth and requiring a minimum number of samples in each leaf.

### Best Parameters

The GridSearchCV process selected:

```text
max_depth = 4
min_samples_leaf = 20
```

### Best Cross-Validation F1-Score

```text
0.647907854484464
```

---

## Model Evaluation

The tuned Decision Tree was evaluated on the unseen test dataset.

### Test Results

| Metric            |                  Score |
| ----------------- | ---------------------: |
| **Test F1-Score** | **0.6586586586586587** |
| **Accuracy**      | **0.8617193836171938** |

### Classification Report

| Class            | Precision |   Recall | F1-Score | Support |
| ---------------- | --------: | -------: | -------: | ------: |
| 0                |      0.97 |     0.86 |     0.91 |    2084 |
| 1                |      0.53 |     0.86 |     0.66 |     382 |
| **Macro Avg**    |  **0.75** | **0.86** | **0.79** |    2466 |
| **Weighted Avg** |  **0.90** | **0.86** | **0.87** |    2466 |

### Confusion Matrix

```text
[[1796  288]
 [  53  329]]
```

This means:

* **1796** → correctly predicted non-purchase sessions
* **288** → non-purchase sessions predicted as purchases
* **53** → purchase sessions predicted as non-purchases
* **329** → correctly predicted purchase sessions

---

## F1-Score Benchmark

The assignment specified an F1-score benchmark of **0.55**.

The final test F1-score obtained was:

```text
0.6586586586586587
```

The tuned model therefore achieved an F1-score above the specified benchmark.

---

## Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* NumPy
* Scikit-learn
* Jupyter Notebook

### Scikit-learn Components

* `train_test_split`
* `ColumnTransformer`
* `StandardScaler`
* `OneHotEncoder`
* `Pipeline`
* `DecisionTreeClassifier`
* `GridSearchCV`
* `f1_score`
* `classification_report`
* `confusion_matrix`
* `accuracy_score`

---

## Project Structure

```text
ML-Model-For-Ecommerce-Purchase-Prediction/
│
├── shop-smart-ecommerce-prediction.ipynb
├── README.md
└── dataset/
    └── shop_smart_ecommerce.csv
```

---

## Key Learning Outcomes

Through this project, I learned how to:

* Prepare numerical and categorical data for machine learning.
* Use `ColumnTransformer` and `Pipeline` for organized preprocessing.
* Handle imbalanced classification problems using class weights.
* Build and tune a Decision Tree Classifier.
* Control Decision Tree complexity using `max_depth` and `min_samples_leaf`.
* Use GridSearchCV with cross-validation.
* Select F1-score as an important metric for imbalanced classification.
* Interpret classification reports and confusion matrices.
* Evaluate a model on unseen test data.

---

## Author

**Parth Jaiswal**<br>
B.Tech CSE — AI/ML & Robotics    
