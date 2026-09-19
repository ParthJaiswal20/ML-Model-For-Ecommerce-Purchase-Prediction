Decision Tree-based ML model for predicting e-commerce purchase behavior.
# ShopSmart E-commerce Purchase Prediction 

## Overview

This project focuses on predicting whether an e-commerce visitor will complete a purchase based on their browsing and session behavior.

The project uses a **Decision Tree Classifier** with preprocessing, class balancing, hyperparameter tuning, and pruning-related complexity control to handle the imbalanced classification problem.

---

## Objective

The main objectives of this project are to:

* Analyze e-commerce session data.
* Perform preprocessing and feature transformations.
* Handle numerical and categorical features appropriately.
* Build a Decision Tree classification model.
* Address class imbalance using class weighting.
* Use hyperparameter tuning to improve model performance.
* Evaluate the model primarily using the **F1-score**.

The target variable is **`Revenue`**, which indicates whether the visitor completed a purchase.

---

## Dataset

The dataset contains **12,330 individual user sessions** collected over a one-year period.

### Features

The dataset contains numerical and categorical session-related features, including:

* `Administrative`
* `Administrative_Duration`
* `Informational`
* `Informational_Duration`
* `ProductRelated`
* `ProductRelated_Duration`
* `BounceRates`
* `ExitRates`
* `PageValues`
* `SpecialDay`
* `Month`
* `OperatingSystems`
* `Browser`
* `Region`
* `TrafficType`
* `VisitorType`
* `Weekend`

### Target

* **`Revenue`** — indicates whether the session resulted in a purchase.

---

## Machine Learning Approach

### 1. Feature Separation

The target variable `Revenue` is separated from the input features.

```python
X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)
```

### 2. Numerical and Categorical Features

Features are automatically separated based on their data types.

* Numerical features → `int64`, `float64`
* Categorical features → `object`, `category`

### 3. Train-Test Split

The dataset is divided into:

* **80% Training data**
* **20% Testing data**

A stratified split is used to maintain the class distribution.

```python
train_test_split(
    X, y,
    random_state=32,
    test_size=0.2,
    stratify=y
)
```

### 4. Feature Preprocessing

#### Numerical Features

`StandardScaler` is used to standardize numerical features.

#### Categorical Features

`OneHotEncoder` converts categorical variables into numerical representations.

```python
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing steps are combined using `ColumnTransformer`.

---

## Model

### Decision Tree Classifier

The main machine learning algorithm used in this project is a **Decision Tree Classifier**.

The model uses:

* `class_weight="balanced"` to address class imbalance.
* `max_depth` to control tree complexity.
* `min_samples_leaf` to prevent overly complex branches.

```python
DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42
)
```

---

## Hyperparameter Tuning

`GridSearchCV` is used to find suitable Decision Tree parameters.

The following parameters are tested:

| Parameter          | Values     |
| ------------------ | ---------- |
| `max_depth`        | 4, 6, 8    |
| `min_samples_leaf` | 20, 30, 50 |

The model uses **5-fold cross-validation** and **F1-score** as the scoring metric.

```python
GridSearchCV(
    pipe,
    param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1
)
```

This also provides pruning-related **tree complexity control** through `max_depth` and `min_samples_leaf`.

---

## Evaluation

Because the dataset is imbalanced, the **F1-score** is the primary evaluation metric.

The project evaluates the final model using:

* F1-score
* Accuracy
* Precision
* Recall
* Classification Report
* Confusion Matrix

The assignment uses an **F1-score benchmark of 0.55**.

### Model Results

Final results will be added after evaluating the tuned model:

* Best Cross-Validation F1: **Not provided yet**
* Best Parameters: **Not provided yet**
* Test F1-score: **Not provided yet**
* Accuracy: **Not provided yet**
* Precision: **Not provided yet**
* Recall: **Not provided yet**
* Confusion Matrix: **Not provided yet**

---

## Workflow

```text
E-commerce Session Dataset
          ↓
Feature & Target Separation
          ↓
Numerical / Categorical Feature Identification
          ↓
Stratified Train-Test Split
          ↓
Feature Preprocessing
   ├── StandardScaler
   └── OneHotEncoder
          ↓
Decision Tree Classifier
          ↓
Class Balancing
          ↓
GridSearchCV + 5-Fold Cross-Validation
          ↓
Hyperparameter Tuning
          ↓
Complexity Control / Pruning
          ↓
Final Prediction
          ↓
F1-Score + Classification Metrics
```

---

## Technologies Used

* **Python**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

  * Decision Tree
  * ColumnTransformer
  * StandardScaler
  * OneHotEncoder
  * Pipeline
  * GridSearchCV
  * Classification Metrics
* **Jupyter Notebook**

---

## Project Structure

```text
ML-Model-For-Ecommerce-Purchase-Prediction/
│
├── shop-smart-ecommerce-purchase-prediction.ipynb
├── README.md
└── dataset/
    └── shop_smart_ecommerce.csv
```

---

## Key Learning Outcomes

* Understanding binary classification for e-commerce prediction.
* Working with numerical and categorical features together.
* Applying feature scaling and one-hot encoding.
* Building a Decision Tree classification model.
* Handling imbalanced classification using class weights.
* Understanding hyperparameter tuning with `GridSearchCV`.
* Using F1-score for imbalanced classification problems.
* Controlling Decision Tree complexity using `max_depth` and `min_samples_leaf`.
* Evaluating classification models using multiple performance metrics.

---

## Author

**Parth Jaiswal**
B.Tech CSE — AI/ML & Robotics
