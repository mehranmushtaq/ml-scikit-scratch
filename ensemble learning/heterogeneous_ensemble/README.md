# Heterogeneous Ensemble Methods

> Combining multiple model types using Voting and Stacking ensembles — covering both **regression** (Power Plant Energy Output) and **classification** (Breast Cancer Diagnosis).

-----

## Overview

This folder demonstrates **heterogeneous ensemble learning**, where fundamentally different model architectures are combined to improve predictive performance. Two ensemble strategies are explored across two tasks:

- **Voting** — aggregates predictions from multiple base models (soft voting for classification, weighted averaging for regression)
- **Stacking** — trains base models and feeds their outputs to a meta-learner via cross-validation

-----

## Notebooks

| Notebook | Task | Dataset | Models |
|---|---|---|---|
| `ml_comparison.ipynb` | Regression | Power Plant Energy Output | Linear Regression, Random Forest, SVR, Voting, Stacking |
| `heterogeneous_ensemble_methods.ipynb` | Classification | Breast Cancer Wisconsin | Logistic Regression, SVC, Decision Tree, Voting, Stacking |

-----

## 📊 ML Comparison — Regression (`ml_comparison.ipynb`)

### Dataset

**Combined Cycle Power Plant (CCPP)**

- **Source:** `powerplant_data.csv`
- **Features:** AT (Ambient Temperature), V (Exhaust Vacuum), AP (Ambient Pressure), RH (Relative Humidity)
- **Target:** PE (Net hourly electrical energy output)
- **Task:** Regression
- **Split:** 67% train / 33% test (`random_state=42`)

### Model Results

| Model | R² Score | Mean CV R² (5-fold) |
|---|---|---|
| Linear Regression | 0.9303 | 0.9285 |
| Random Forest Regressor | 0.9645 | 0.9640 |
| Support Vector Regressor (SVR) | 0.9481 | 0.9476 |
| **Voting Regressor** (LR+RF+SVR, weights=[1,2,1]) | **0.9589** | **0.9580** |
| **Stacking Regressor** (meta: Ridge) | **0.9648** | **0.9644** |

> **Key finding:** Stacking (LR + RF + SVR → Ridge meta-learner) achieves the highest R² of **0.9648**, narrowly outperforming even the best individual model (Random Forest at 0.9645), while the Voting Regressor delivers a strong balanced performance by up-weighting Random Forest.

### Model Configurations

**Random Forest** (tuned via GridSearchCV):
```python
RandomForestRegressor(
    n_estimators=200, max_depth=20,
    min_samples_split=3, min_samples_leaf=2,
    max_features='sqrt', random_state=42
)
```

**SVR** (tuned via GridSearchCV):
```python
Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=100, epsilon=1, gamma=0.5))
])
```

**Voting Regressor:**
```python
VotingRegressor(
    estimators=[('lr', pipe_lr), ('rf', best_rf), ('svr', best_svr)],
    weights=[1, 2, 1]
)
```

**Stacking Regressor:**
```python
StackingRegressor(
    estimators=[('lr', pipe_lr), ('rf', best_rf), ('svr', best_svr)],
    final_estimator=Ridge(),
    cv=5
)
```

-----

## 🔬 Heterogeneous Ensemble Methods — Classification (`heterogeneous_ensemble_methods.ipynb`)

### Dataset

**Breast Cancer Wisconsin (Diagnostic)**

- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Samples:** 569 total — 357 malignant (class 1), 212 benign (class 0)
- **Features:** 30 numeric features derived from cell nucleus images
- **Task:** Binary classification
- **Split:** 70% train / 30% test (`random_state=42`)

### Results

#### Voting Classifier

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 | 0.97 | 0.98 | 0.98 | 63 |
| 1 | 0.99 | 0.98 | 0.99 | 108 |
| **Accuracy** | | | **0.98** | **171** |
| Macro Avg | 0.98 | 0.98 | 0.98 | 171 |
| Weighted Avg | 0.98 | 0.98 | 0.98 | 171 |

#### Stacking Classifier

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 | 0.97 | 0.98 | 0.98 | 63 |
| 1 | 0.99 | 0.98 | 0.99 | 108 |
| **Accuracy** | | | **0.98** | **171** |
| Macro Avg | 0.98 | 0.98 | 0.98 | 171 |
| Weighted Avg | 0.98 | 0.98 | 0.98 | 171 |

**Stacking AUC-ROC Score: 0.9975**

#### Stacking Confusion Matrix

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 62 | 1 |
| **Actual 1** | 2 | 106 |

### Model Configurations

**Base Estimators:**

| Model | Details |
|---|---|
| `LogisticRegression` | Wrapped in `StandardScaler` pipeline |
| `SVC` | Wrapped in `StandardScaler` pipeline, `probability=True` |
| `DecisionTreeClassifier` | `max_depth=3` |

**Voting:**
```python
VotingClassifier(
    estimators=[("lr", lr_scaled), ("svc", svc_scaled), ("dtc", dtc)],
    voting='soft'
)
```

**Stacking:**
```python
StackingClassifier(
    estimators=[("lr", lr_scaled), ("svc", svc_scaled), ("dtc", dtc)],
    final_estimator=LogisticRegression(),
    cv=5
)
```

-----

## Project Structure

```
heterogeneous_ensemble/
├── ml_comparison.ipynb                      # Regression comparison (Power Plant)
├── heterogeneous_ensemble_methods.ipynb     # Classification ensembles (Breast Cancer)
├── powerplant_data.csv                      # Dataset for regression notebook
└── README.md
```

-----

## Requirements

```
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install via:

```bash
pip install scikit-learn pandas matplotlib seaborn
```

-----

## Usage

```bash
# Clone the repository
git clone https://github.com/mehranmushtaq/ml-scikit-scratch.git
cd ml-scikit-scratch/ensemble\ learning/heterogeneous_ensemble

# Open either notebook
jupyter notebook ml_comparison.ipynb
jupyter notebook heterogeneous_ensemble_methods.ipynb
```

-----

## Key Takeaways

- **Stacking consistently wins** — by learning how to best combine model outputs via a meta-learner, it edges out even the strongest individual model (Random Forest) on the regression task
- **Voting with weights** is a simpler but effective alternative, especially when one model (Random Forest) is clearly stronger — up-weighting it improves ensemble performance
- **Soft voting** (averaging predicted probabilities) generally outperforms hard voting on classification tasks
- **Scaling is critical** for distance-based models (LR, SVM) — pipelines ensure no data leakage
- **GridSearchCV tuning** on base models before ensembling gives a much stronger starting point than default hyperparameters

-----

## References

- [scikit-learn: VotingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html)
- [scikit-learn: StackingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
- [scikit-learn: VotingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)
- [scikit-learn: StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
- [Combined Cycle Power Plant Dataset](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)
- [Breast Cancer Wisconsin Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset)

-----

*Part of the [ml-scikit-scratch](https://github.com/mehranmushtaq/ml-scikit-scratch) repository by [@mehranmushtaq](https://github.com/mehranmushtaq).*
