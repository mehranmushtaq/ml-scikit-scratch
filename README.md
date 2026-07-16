## 🧠Machine Learning with Scikit-Learn & From Scratch

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-189AB4?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.23+-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

### *End-to-End Machine Learning Repository — From Theory to Deployment*

> 🏆 **94.89% accuracy** on Disease Prediction  |  📈 **F1: 0.6485** on E-Commerce (+18% above benchmark)  |  🛒 **4 customer segments** uncovered in SmartCart  |  🧬 **89% Precision@K** on Thyroid Outlier Detection  |  🚀 **2 Projects Deployed to Production**

-----

> **Mehran Mushtaq** · Data Science & Machine Learning Track · Python · Scikit-Learn

-----

## What Is This Repository?

This isn’t just a collection of notebooks.

It’s a **complete learning journey** — from implementing algorithms by hand to deploying production-ready ML pipelines — built over the course of an intensive internship.

Every folder tells a chapter of that story: understanding *why* an algorithm works before trusting a library to do it. Writing clean pipelines. Tuning models properly. Building real projects that solve real problems — and then **shipping them**.

-----

## What Makes This Different

|Approach                 |What Was Done                                                                    |
|-------------------------|---------------------------------------------------------------------------------|
|**From Scratch**         |Core algorithms implemented in pure Python/NumPy — no sklearn shortcuts          |
|**Pipelines**            |Clean, reproducible workflows using `sklearn.Pipeline` to prevent data leakage   |
|**GridSearchCV**         |Systematic hyperparameter tuning on real datasets                                |
|**Cross-Validation**     |Robust evaluation using k-fold CV with stratification                            |
|**Unsupervised Learning**|Clustering, anomaly detection, and dimensionality reduction on real data         |
|**End-to-End Projects**  |Full pipelines from raw data → feature engineering → model → evaluation → insight|
|**🚀 Deployed**           |Projects shipped beyond notebooks — live demos accessible online                 |

-----

## 🚀 Live Deployments

> These projects have been fully deployed and are accessible as interactive web applications.

|Project                   |Domain     |Live Demo                                                                                                                                                                                                                                                                                  |
|--------------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|🏠 House Price Prediction  |Real Estate|[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://house-prices-ml.streamlit.app/)                                                                                                                          |
|💳 CreditWise Loan Approval|Finance    |[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://creditwise-loan.streamlit.app/) · [![Repo](https://img.shields.io/badge/GitHub-Repo-181717?style=flat-square&logo=github)](https://github.com/mehranmushtaq/creditwise-loan-system)|


-----

## Projects At a Glance

|Project                                                                  |Domain     |Technique                              |Key Result                    |Status    |
|-------------------------------------------------------------------------|-----------|---------------------------------------|------------------------------|----------|
|[🏠 House Price Prediction](#-house-price-prediction--deployed)           |Real Estate|Linear Regression + Feature Engineering|Deployed to production        |🚀 **Live**|
|[💳 CreditWise Loan Approval](#-creditwise-loan-approval-system--deployed)|Finance    |Classification Pipeline                |Deployed to production        |🚀 **Live**|
|[🏥 Disease Prediction Pipeline](#-disease-prediction-pipeline)           |Healthcare |Voting Classifier Ensemble             |94.89% accuracy               |✅ Complete|
|[🛒 E-Commerce Purchase Prediction](#-e-commerce-purchase-prediction)     |Retail     |Decision Tree                          |F1: 0.6485 (+18% benchmark)   |✅ Complete|
|[🔬 SmartCart Customer Segmentation](#-smartcart-customer-segmentation)   |Retail     |K-Means + Agglomerative + PCA          |4 actionable customer personas|✅ Complete|
|[🧬 Thyroid Outlier Detection](#-thyroid-outlier-detection)               |Healthcare |Isolation Forest + LOF                 |89% Precision@K               |✅ Complete|

-----

## 🗂️ Repository Structure

```
ml-scikit-scratch/
│
├── 📁 Datasets/                              # Shared datasets across experiments
│   ├── Emotion_classify_Data.csv
│   ├── Iris.csv
│   ├── Social_Network_Ads.csv
│   ├── house_prices_practice.csv
│   └── insurance.csv
│
├── 📁 linear_regression/
│   ├── Linear_regression.ipynb
│   └── README.md
│
├── 📁 Logistic Regression/
│   ├── Logistic_Regressor.ipynb
│   └── README.md
│
├── 📁 KNN/
│   ├── Knn.ipynb
│   └── README.md
│
├── 📁 Decision tree/
│   ├── decision_tree_classifier.ipynb
│   ├── decision_tree_regressor.ipynb
│   └── README.md
│
├── 📁 Naive bayes/
│   ├── naive_bayes.ipynb
│   └── README.md
│
├── 📁 Support Vector Machine/
│   ├── svc.ipynb
│   ├── svr.ipynb
│   └── README.md
│
├── 📁 Regularizaton(Lasso:Ridge)/
│   ├── lasso_ridge.ipynb
│   └── README.md
│
├── 📁 ensemble learning/
│   ├── bagging/                              # Random Forest
│   │   ├── Random_forest.ipynb
│   │   └── README.md
│   └── boosting/                             # AdaBoost, Gradient Boosting, XGBoost
│       ├── ada_boosting.ipynb
│       ├── gradient_boosting.ipynb
│       ├── xgboost.ipynb
│       └── README.md
│
├── 📁 ml-from-scratch/                      
│   ├── linear_reg.ipynb
│   ├── logistic_reg.ipynb
│   └── knn_regressor.ipynb
│
├── 📁 unsupervised ml/
│   ├── k_means.ipynb
│   ├── hiearchichal_clustering.ipynb
│   ├── dbscan.ipynb
│   └── README.md
│
├── 📁 Projects/                              
│   ├── 📁 house-price-prediction/            #  DEPLOYED
│   ├── 📁 customer-segmentation-smartcart/
│   ├── 📁 CreditWise Loan System/            #  DEPLOYED (separate repo)
│   ├── 📁 ecommerce-purchase-prediction/
│   ├── 📁 thyroid_outlier_detection/
│   └── 📁 disease_prediction_pipeline/
│
├── notebook_vs_production.md
├── requirements.txt
└── README.md
```

-----

## What This Repository Covers

### Supervised Learning

- **[Linear Regression](./linear_regression)** — Predicting continuous outcomes; from OLS to regularised versions
- **[Logistic Regression](./Logistic%20Regression)** — Binary and multi-class classification with polynomial features
- **[K-Nearest Neighbours (KNN)](./KNN)** — Distance-based classification and regression
- **[Decision Trees](./Decision%20tree)** — Classifier & regressor with pre/post pruning strategies
- **[Naive Bayes](./Naive%20bayes)** — Probabilistic classification with Gaussian distributions
- **[Support Vector Machines](./Support%20Vector%20Machine)** — SVC for classification, SVR for regression
- **[Regularization](./Regularizaton(Lasso:Ridge))** — Lasso (L1) and Ridge (L2) to control overfitting

### Ensemble Methods

- **[Bagging](./ensemble%20learning/bagging)** — Random Forest with feature importance analysis
- **[Boosting](./ensemble%20learning/boosting)** — AdaBoost, Gradient Boosting, and XGBoost

### Unsupervised Learning

- **[K-Means Clustering](./unsupervised%20ml)** — Centroid-based segmentation with elbow & silhouette analysis
- **[Hierarchical Clustering](./unsupervised%20ml)** — Agglomerative dendrograms with Ward linkage
- **[DBSCAN](./unsupervised%20ml)** — Density-based, anomaly-resistant clustering

### ML From Scratch (Pure Python / NumPy)

- **[Linear Regression](./ml-from-scratch)** — Gradient descent, cost function, weight updates
- **[Logistic Regression](./ml-from-scratch)** — Sigmoid, binary cross-entropy, manual backprop
- **[KNN Regressor](./ml-from-scratch)** — Euclidean distance, neighbour voting

-----

## Featured Projects

-----

### 🏠 House Price Prediction — *Deployed*

> **Domain:** Real Estate · **Type:** Supervised Regression · **Dataset:** `house_prices_practice.csv` · 🚀 [**Live Demo**](https://house-prices-ml.streamlit.app/)

**The Problem:** Predict residential property sale prices from structural and neighbourhood features — a classic, high-value regression problem.

**Approach & Key Decisions:**

- **Exploratory Data Analysis:** Identified key drivers of price — GrLivArea, OverallQual, TotalBsmtSF, and neighbourhood — through correlation analysis and visualisations
- **Feature Engineering:** Created interaction features (e.g. total floor area), handled missing values with domain-aware imputation, and log-transformed the skewed target variable to improve model fit
- **Preprocessing Pipeline:** `ColumnTransformer` applied scaling to numerical features and one-hot encoding to categorical ones inside a single `sklearn.Pipeline`
- **Model Selection:** Evaluated Linear Regression, Ridge, and Lasso — tuned regularisation strength via `GridSearchCV`
- **Deployment:** Exported the trained pipeline and built an interactive **Streamlit** web app so anyone can input house features and receive an instant price estimate

**Highlights:**

|Aspect       |Detail                                                                |
|-------------|----------------------------------------------------------------------|
|Target       |Sale price (log-transformed)                                          |
|Key Features |Living area, overall quality, basement size, year built, neighbourhood|
|Preprocessing|Median imputation, OneHot encoding, StandardScaler                    |
|Deployment   |Streamlit web application                                             |

**Tech:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn` (Pipeline, Ridge, Lasso, GridSearchCV), `streamlit`

-----

### 💳 CreditWise Loan Approval System — *Deployed*

> **Domain:** Finance · **Type:** Supervised Classification · **Deliverable:** Notebook + Production `.py` script + Live App · 🚀 [**Live Demo**](https://creditwise-loan.streamlit.app/) · 📁 [**Repository**](https://github.com/mehranmushtaq/creditwise-loan-system)

**The Problem:** Manual loan approval is slow, inconsistent, and prone to bias. CreditWise needed an automated classification system to assess applicant risk from financial and demographic data — consistently, at scale.

**Approach & Key Decisions:**

- **Feature Engineering:** Derived risk signals — debt-to-income proxies, employment stability indicators, and interaction terms between credit history and income band
- **Preprocessing Pipeline:** `ColumnTransformer` applied different strategies to numerical (imputation + scaling) and categorical (encoding) features simultaneously
- **Model Selection:** Logistic Regression, Random Forest, and Gradient Boosting evaluated; best model tuned with `GridSearchCV`
- **Production Delivery:** Final model exported as a standalone `.py` script and deployed as an interactive **Streamlit** application — accessible via the live demo link above

**Highlights:**

|Aspect        |Detail                                                                    |
|--------------|--------------------------------------------------------------------------|
|Input features|Income, loan amount, credit history, employment, dependents, property area|
|Preprocessing |Median imputation, OneHot encoding, StandardScaler                        |
|Evaluation    |Accuracy, Precision, Recall, F1, Confusion Matrix                         |
|Deployment    |Streamlit web application (separate repo)                                 |

**Tech:** `pandas`, `sklearn` (Pipeline, ColumnTransformer, GridSearchCV, RandomForestClassifier, GradientBoostingClassifier, LogisticRegression), `matplotlib`, `streamlit`

-----

### 🏥 Disease Prediction Pipeline

> **Domain:** Healthcare · **Type:** Supervised Classification · **Dataset:** 9,800 patients (NovaGen Research Labs)

**The Problem:** NovaGen Research Labs needed a reliable automated system to classify patients as healthy or at risk based on clinical features — to streamline clinical trial participant selection and reduce manual screening time.

**Approach & Key Decisions:**

- **Pipeline-first design:** Entire workflow (imputation → encoding → scaling → model) encapsulated in a single `sklearn.Pipeline` — zero data leakage guaranteed
- **Ensemble strategy:** Hard-voting classifier combining Logistic Regression, Random Forest, and Naïve Bayes — majority vote reduces the variance of any single model
- **Stratified K-Fold CV:** Class proportions preserved across every fold — critical for a medical dataset
- **Evaluation focus:** Optimised for **recall** alongside accuracy — in clinical settings, false negatives are the higher-risk failure mode

**Results:**

|Metric             |Score                              |
|-------------------|-----------------------------------|
|Accuracy           |**94.89%**                         |
|CV Recall          |**95.46%**                         |
|Validation Strategy|Stratified K-Fold Cross-Validation |
|Model              |Hard-Voting Ensemble (LR + RF + NB)|

**Tech:** `pandas`, `sklearn` (Pipeline, VotingClassifier, LogisticRegression, RandomForestClassifier, GaussianNB, StratifiedKFold, cross_val_score)

-----

### 🛒 E-Commerce Purchase Prediction

> **Domain:** Retail · **Type:** Supervised Classification · **Dataset:** 12,330 browsing sessions (ShopSmart)

**The Problem:** Only ~15% of ShopSmart’s 12,330 recorded browsing sessions ended in a purchase. The marketing team needed a model to identify which visitors were likely to convert.

**Approach & Key Decisions:**

- **Class imbalance handling:** `class_weight='balanced'` in the Decision Tree forces attention on the minority (purchase) class
- **Depth pruning:** `max_depth` tuned via GridSearchCV to prevent overfitting on the majority class
- **Evaluation:** Prioritised **F1-Score** as the primary metric — the harmonic mean of precision and recall

**Results:**

|Metric         |Score                          |
|---------------|-------------------------------|
|F1-Score       |**0.6485**                     |
|Benchmark F1   |0.55                           |
|Improvement    |**+18% above benchmark**       |
|Class Imbalance|85% non-purchase / 15% purchase|

**Tech:** `pandas`, `sklearn` (DecisionTreeClassifier, GridSearchCV, classification_report, train_test_split), `matplotlib`, `seaborn`

-----

### 🔬 SmartCart Customer Segmentation

> **Domain:** Retail · **Type:** Unsupervised Learning · **Dataset:** 2,240 customers × 22 features

**The Problem:** SmartCart was treating all customers the same — one message, one offer, one strategy. The goal was to discover natural groupings so marketing, product, and retention teams could act with precision.

**Approach & Key Decisions:**

- **Feature engineering:** `Age`, `Customer_Tenure_Days`, `Total_Spending`, `Total_Children` derived from raw columns
- **Outlier removal:** Customers aged 90+ and income above $600k dropped
- **Dimensionality reduction:** PCA reduced the feature space to 4 principal components (~55% variance explained)
- **K selection:** Elbow Method + Silhouette Score both converged on **k = 4**
- **Final model:** Agglomerative Clustering (Ward linkage) selected over K-Means for deterministic, tighter cluster boundaries

**Results — Discovered Customer Segments:**

|Cluster|Avg Income|Avg Spending|Profile                          |
|-------|----------|------------|---------------------------------|
|0      |~$42,706  |~$327       |Budget-conscious, larger families|
|1      |~$66,279  |~$1,055     |Affluent loyalists, catalog-heavy|
|2      |~$35,326  |~$110       |Low-income, high web visits      |
|3      |~$74,727  |~$1,271     |Premium spenders, fewest children|

**Tech:** `pandas`, `numpy`, `seaborn`, `matplotlib`, `sklearn` (PCA, KMeans, AgglomerativeClustering, silhouette_score), `kneed`

-----

### 🧬 Thyroid Outlier Detection

> **Domain:** Healthcare · **Type:** Unsupervised Anomaly Detection · **Dataset:** 1,000 patient lab records

**The Problem:** In 1,000 thyroid hormone lab results, a small number of patients had clinically abnormal profiles — but no labels indicated which ones. The system had to find anomalies without any ground truth.

**Approach & Key Decisions:**

- **Dual-detector approach:** Isolation Forest (global anomalies) + LOF (local outliers) — agreement between both increases confidence in flagged cases
- **No labels required:** System learns the “normal” distribution from the data itself
- **Validation:** Flagged patients cross-checked against clinical reference ranges for hypo/hyperthyroid conditions

**Results:**

|Metric         |Score                    |
|---------------|-------------------------|
|Precision@K    |**89%**                  |
|Detectors Used |Isolation Forest + LOF   |
|Labels Required|None (fully unsupervised)|

**Tech:** `pandas`, `numpy`, `sklearn` (IsolationForest, LocalOutlierFactor, StandardScaler), `matplotlib`, `seaborn`

-----

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/mehranmushtaq/ml-scikit-scratch.git

# Navigate into the repo
cd ml-scikit-scratch

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

-----

## Key Concepts Practiced

```
Data Preprocessing & Feature Engineering
Encoding (LabelEncoder, One-Hot, ColumnTransformer)
Train/Test Split with Stratification
StandardScaler inside Pipelines (zero data leakage)
GridSearchCV for Hyperparameter Tuning
Cross-Validation (k-fold, stratified)
Class Imbalance Handling (class_weight, resampling)
Dimensionality Reduction via PCA
Unsupervised Validation (Elbow Method, Silhouette Score)
Anomaly Detection (Isolation Forest, LOF)
Model Evaluation (Accuracy, F1, Precision, Recall, AUC)
Confusion Matrix Analysis
Feature Importance Visualisation
Algorithms Implemented From Scratch (NumPy only)
Model Deployment (Streamlit)
```

-----

## Author

**Mehran Mushtaq** · Data Science & Machine Learning Track

[![GitHub](https://img.shields.io/badge/GitHub-mehranmushtaq-181717?style=flat-square&logo=github)](https://github.com/mehranmushtaq)

-----

*“First, solve the problem. Then, write the code.”* — John Johnson

