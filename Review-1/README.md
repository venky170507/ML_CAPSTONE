# Machine Learning Capstone Project — Review 1

## 23CSE301 — Machine Learning

This repository contains the **Review 1 Regression Track** implementation for the
23CSE301 Machine Learning Capstone Project.

The implementation follows the course requirement that all ten regression
algorithms are trained using the same Ames Housing dataset and the same held-out
test split.

---

## Project Scope

### Review 1 — Regression Track

The regression problem is:

> **Predict the sale price of a residential property from its property
> characteristics.**

### Dataset

**Ames Housing Dataset**

File:

```text
Review-1/data/AmesHousing.csv
```

Target variable:

```text
SalePrice
```

Dataset size:

```text
2930 rows × 82 columns
```

The dataset contains numerical and categorical property attributes, including
living area, overall quality, year built, basement characteristics, garage
information, neighborhood, and other residential-property features.

---

# Required Regression Algorithms

The following ten algorithms are implemented in **ten separate Jupyter
notebooks**:

| # | Algorithm | Notebook |
|---|---|---|
| 1 | Linear Regression | `01_linear_regression.ipynb` |
| 2 | Ridge Regression | `02_ridge_regression.ipynb` |
| 3 | Lasso Regression | `03_lasso_regression.ipynb` |
| 4 | ElasticNet Regression | `04_elasticnet_regression.ipynb` |
| 5 | Polynomial Regression | `05_polynomial_regression.ipynb` |
| 6 | Decision Tree Regressor | `06_decision_tree_regressor.ipynb` |
| 7 | Random Forest Regressor | `07_random_forest_regressor.ipynb` |
| 8 | Gradient Boosting Regressor | `08_gradient_boosting_regressor.ipynb` |
| 9 | Support Vector Regressor (SVR) | `09_svr.ipynb` |
| 10 | K-Nearest Neighbors Regressor | `10_knn_regressor.ipynb` |

An additional notebook consolidates the results:

```text
00_regression_model_comparison.ipynb
```

---

# Regression Workflow

```text
Ames Housing Dataset
        ↓
Dataset Loading
        ↓
Dataset Audit
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Missing-Value Handling
        ↓
Outlier Checking/Treatment
        ↓
Categorical Encoding
        ↓
Feature Scaling where required
        ↓
Regression Model
        ↓
R² / RMSE / MAE
        ↓
Model Comparison
        ↓
Hyperparameter Tuning
        ↓
5-Fold CV for Top 2 Models
        ↓
Final Visualisations
```

---

# Dataset Audit and EDA

Each algorithm notebook contains the common dataset audit and EDA required for
Review 1.

The audit reports:

- Dataset shape
- Data types
- Missing-value counts
- Duplicate rows
- Target distribution
- Number of numerical features
- Number of categorical features

EDA includes:

- Numerical feature distribution plots
- Target distribution
- Correlation heatmap
- Overall Quality vs SalePrice scatter plot
- Gr Liv Area vs SalePrice scatter plot

Each notebook also contains Markdown sections where the team must write its
own observations after inspecting the plots.

---

# Preprocessing

The preprocessing pipeline is designed to avoid data leakage.

### Numerical features

- Median imputation
- Standard scaling for scale-sensitive algorithms

### Categorical features

- Most-frequent imputation
- One-hot encoding
- Unknown categories are handled safely

### Train/Test Split

```text
80% Training
20% Testing
```

The split uses:

```text
random_state = 42
```

The same split is used by all ten regression algorithms so their test-set
metrics can be compared fairly.

### Outlier Treatment

Extreme `Gr Liv Area` values are checked using a 3×IQR upper threshold learned
from the training data only. The held-out test set is not filtered using this
threshold.

### Feature Engineering

The notebooks create predictor-only derived features:

```text
TotalSF
= Total Bsmt SF + 1st Flr SF + 2nd Flr SF
```

and

```text
TotalBathrooms
= Full Bath
+ 0.5 × Half Bath
+ Bsmt Full Bath
+ 0.5 × Bsmt Half Bath
```

These transformations use only predictor variables and do not use `SalePrice`.

The team should review these features and provide its own justification in the
final notebook/report.

---

# Model Evaluation

Every regression model is evaluated on the same held-out test set using:

- **R² Score**
- **RMSE — Root Mean Squared Error**
- **MAE — Mean Absolute Error**

The comparison notebook produces a single table ranked by R².

Example structure:

| Model | R² | RMSE | MAE |
|---|---:|---:|---:|
| Model 1 | — | — | — |
| Model 2 | — | — | — |
| Model 3 | — | — | — |
| Model 4 | — | — | — |
| Model 5 | — | — | — |
| Model 6 | — | — | — |
| Model 7 | — | — | — |
| Model 8 | — | — | — |
| Model 9 | — | — | — |
| Model 10 | — | — | — |

The actual values will be generated after the notebooks are executed.

---

# Hyperparameter Tuning

The Review 1 requirement is satisfied by tuning at least two regression
models.

Current implementations include:

### Ridge Regression

`GridSearchCV` is used to tune:

```text
alpha
```

### Random Forest Regressor

`RandomizedSearchCV` is used to tune:

```text
n_estimators
max_depth
min_samples_split
max_features
```

The best parameters and cross-validation score are displayed in the
corresponding notebooks.

---

# Cross-Validation

The two best-performing models are identified from the common test-set
comparison.

The comparison notebook then performs:

```text
5-Fold Cross-Validated R²
```

for those two models.

The five fold scores, mean, and standard deviation are reported.

---

# Required Visualisations

The project includes code for:

- Feature distribution plots
- Target distribution
- Correlation heatmap
- Feature-target scatter plots
- Predicted vs Actual plot
- Residual plot
- Tree-based feature importance plot

The comparison notebook generates the final predicted-vs-actual and residual
plots for the best-performing model and a feature-importance plot for the best
available tree-based model.

---

# Repository Structure

```text
MachineLearningCaseStudy/
│
├── README.md
├── requirements.txt
├── .gitignore
│
└── Review-1/
    │
    ├── data/
    │   └── AmesHousing.csv
    │
    ├── notebooks/
    │   ├── 00_regression_model_comparison.ipynb
    │   ├── 01_linear_regression.ipynb
    │   ├── 02_ridge_regression.ipynb
    │   ├── 03_lasso_regression.ipynb
    │   ├── 04_elasticnet_regression.ipynb
    │   ├── 05_polynomial_regression.ipynb
    │   ├── 06_decision_tree_regressor.ipynb
    │   ├── 07_random_forest_regressor.ipynb
    │   ├── 08_gradient_boosting_regressor.ipynb
    │   ├── 09_svr.ipynb
    │   └── 10_knn_regressor.ipynb
    │
    ├── results/
    │   └── Generated after notebooks are executed
    │
    ├── models/
    │   └── Generated fitted model files
    │
    └── src/
        └── preprocessing.py
```

---

# Environment Setup

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd MachineLearningCaseStudy
```

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Start Jupyter

```bash
jupyter notebook
```

Open:

```text
Review-1/notebooks/
```

---

# Recommended Execution Order

Run the algorithm notebooks first:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10
```

Then run:

```text
00_regression_model_comparison.ipynb
```

The comparison notebook reads the result files generated by the ten algorithm
notebooks.

---

# Reproducibility

The project uses:

```text
random_state = 42
```

where applicable.

All algorithms use the same held-out test split.

Data-driven preprocessing operations such as imputation, encoding, scaling,
and the outlier threshold are learned from training data only.

This prevents information from the held-out test set from leaking into model
training.

---

# Technologies Used

- Python 3
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

# Current Review 1 Progress

- [x] Git repository initialized
- [x] Ames Housing dataset added
- [x] Project structure created
- [x] Regression preprocessing pipeline created
- [x] Linear Regression
- [x] Ridge Regression
- [x] Lasso Regression
- [x] ElasticNet Regression
- [x] Polynomial Regression
- [x] Decision Tree Regressor
- [x] Random Forest Regressor
- [x] Gradient Boosting Regressor
- [x] Support Vector Regressor
- [x] KNN Regressor
- [x] Model comparison notebook
- [ ] Execute all regression notebooks
- [ ] Execute all classification notebooks
- [ ] Review and write team interpretations
- [ ] Finalise regression and classification results tables
- [ ] Finalise Review 1 presentation

---



# Classification Track — Part A

## Problem Statement

Predict whether a person's annual income is greater than **$50K** using
demographic, education, employment, and financial attributes.

### Dataset

**Adult Income Dataset**

The dataset is stored as:

```text
Review-1/data/AdultIncome.csv
```

If the CSV is not present, the classification notebooks automatically retrieve
the Adult dataset through scikit-learn/OpenML.

### Target

```text
income
```

Classes:

```text
<=50K
>50K
```

---

## Required Classification Algorithms

Five separate notebooks implement the required Part-A algorithms:

| # | Algorithm | Notebook |
|---|---|---|
| 1 | Logistic Regression | `01_logistic_regression.ipynb` |
| 2 | K-Nearest Neighbors | `02_knn_classifier.ipynb` |
| 3 | Gaussian Naive Bayes | `03_gaussian_naive_bayes.ipynb` |
| 4 | Decision Tree Classifier | `04_decision_tree_classifier.ipynb` |
| 5 | Support Vector Classifier | `05_svc.ipynb` |

The comparison notebook is:

```text
00_classification_model_comparison.ipynb
```

---

## Classification Workflow

```text
Adult Income Dataset
        ↓
Dataset Audit
        ↓
Exploratory Data Analysis
        ↓
Missing-Value Handling
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
Stratified 80/20 Train/Test Split
        ↓
Classification Model
        ↓
Accuracy + Weighted F1
        ↓
Confusion Matrix
        ↓
Model Comparison
```

Preprocessing is performed inside a scikit-learn Pipeline/ColumnTransformer
so that imputers, encoders, and scalers are fitted using training data only.

---

## Classification Evaluation

Each model reports:

- Accuracy
- Weighted F1 Score
- Confusion Matrix

The comparison notebook combines the five model results into one table and
creates a metric comparison chart.

---

## Classification Repository Structure

```text
Review-1/
├── data/
│   ├── AmesHousing.csv
│   ├── AdultIncome.csv
│   └── README.md
│
├── notebooks/
│   ├── 00_regression_model_comparison.ipynb
│   ├── 01_linear_regression.ipynb
│   ├── 02_ridge_regression.ipynb
│   ├── 03_lasso_regression.ipynb
│   ├── 04_elasticnet_regression.ipynb
│   ├── 05_polynomial_regression.ipynb
│   ├── 06_decision_tree_regressor.ipynb
│   ├── 07_random_forest_regressor.ipynb
│   ├── 08_gradient_boosting_regressor.ipynb
│   ├── 09_svr.ipynb
│   ├── 10_knn_regressor.ipynb
│   │
│   ├── 00_classification_model_comparison.ipynb
│   ├── 01_logistic_regression.ipynb
│   ├── 02_knn_classifier.ipynb
│   ├── 03_gaussian_naive_bayes.ipynb
│   ├── 04_decision_tree_classifier.ipynb
│   └── 05_svc.ipynb
│
├── results/
│   └── classification/
│
└── src/
    ├── preprocessing.py
    ├── classification_preprocessing.py
    └── download_adult_dataset.py
```

---

## Classification Execution Order

Run the five algorithm notebooks:

```text
01 → 02 → 03 → 04 → 05
```

Then run:

```text
00_classification_model_comparison.ipynb
```

The comparison notebook reads the result CSV files produced by the five
algorithm notebooks.


# Academic Integrity and AI Assistance

Generative AI was used for **code scaffolding, notebook structure, and
development assistance**.

The project team is responsible for:

- Reviewing and understanding every line of code
- Verifying all preprocessing decisions
- Reviewing feature-engineering choices
- Running and validating all experiments
- Writing the final data observations and interpretations
- Explaining the algorithms and results during the viva

All analysis, interpretation, and conclusions presented for the case study
must be verified and written by the project team.

---

# Team Members

| Name | Roll Number | Responsibility |
|---|---|---|
| Member 1 | — | Regression Track |
| Member 2 | — | Regression Track |
| Member 3 | — | Regression Track |

---

# References

- 23CSE301 Machine Learning — Capstone Project Guidelines, Algorithm List &
  Evaluation Rubrics
- Ames Housing Dataset
