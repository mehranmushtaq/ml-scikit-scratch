# 🏠 Ames Housing AI — Price Prediction Engine

> **Live Demo →** [house-prices-ml.streamlit.app](https://house-prices-ml.streamlit.app/)

A machine learning web app that predicts residential property prices using the Ames Housing Dataset. Built with XGBoost, scikit-learn, and Streamlit — achieving an **R² of 0.9291** on the test set.

-----

## What It Does

Adjust six property sliders in the sidebar and hit **Predict House Value**. The app instantly renders:

- **Price Gauge** — Plotly indicator showing the market estimate
- **Property DNA Radar** — normalized 5-axis signature (Quality, Space, Modernity, Garage, Baths)
- **Summary Cards** — price per sq ft, estimated annual taxes, monthly mortgage estimate, home age

-----

## Model Performance

|Metric  |Lasso (Baseline)|XGBoost (Champion)|
|--------|----------------|------------------|
|MAE     |$15,033         |**$13,317**       |
|RMSE    |$21,707         |—                 |
|R² Score|0.9192          |**0.9291**        |

### Training Pipeline

```
Raw CSV  →  Null Imputation  →  Outlier Removal  →  One-Hot Encoding (261 features)
         →  StandardScaler  →  XGBRegressor (GridSearchCV, 90 fits, CV=5)
```

**Best XGBoost params:** `n_estimators=1000`, `learning_rate=0.1`, `max_depth=3`

-----

## Top Predictors

```
Overall Qual     ████████████████████  0.22
Gr Liv Area      ████████████████      0.18
Garage Cars      ██████████            0.11
Garage Area      ████████              0.09
Total Bsmt SF    ███████               0.08
1st Flr SF       ██████                0.07
Year Built       █████                 0.06
```

-----

## Tech Stack

|Layer              |Technology                           |
|-------------------|-------------------------------------|
|UI & Hosting       |Streamlit + Streamlit Community Cloud|
|Visualizations     |Plotly (Gauge, Radar)                |
|ML Model           |XGBoost via scikit-learn Pipeline    |
|Data Processing    |pandas, NumPy                        |
|Model Serialization|joblib                               |

-----

## Run Locally

**1. Clone the repo**
```bash
git clone [https://github.com/mehranmushtaq/ml-scikit-scratch.git](https://github.com/mehranmushtaq/ml-scikit-scratch.git)
cd ml-scikit-scratch/Projects/house_prices_prediction
```
**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add model file** *(optional — app runs in demo mode without it)*

```
house-prices-ml/
├── app.py
├── ames_hpousing_xgb_model.pkl    ← your trained model
└── README.md
```

**4. Launch**

```bash
streamlit run app.py
```

Open <http://localhost:8501>

-----

## Project Structure

```
house-prices-ml/
├── app.py                # Streamlit application
├── notebook.ipynb        # Full ML pipeline (EDA → training → evaluation)
├── ames_xgb_model.pkl    # Serialized XGBoost champion model
└── README.md
```

### Notebook Walkthrough

|Cells|Step                                             |
|-----|-------------------------------------------------|
|1–14 |Load data, inspect nulls                         |
|15–21|Impute missing values, strip column names        |
|22–24|Fill remaining nulls (Lot Frontage, Electrical)  |
|25–26|Correlation analysis, outlier removal            |
|28–29|One-hot encoding → 261 features, train/test split|
|36–37|Lasso baseline — MAE $15,033, R² 0.9192          |
|41–42|XGBoost GridSearchCV — MAE $13,317, R² 0.9291    |
|43–44|Feature importance chart, save model             |

-----

## Dataset

**Ames Housing Dataset** — 2,930 residential property sales in Ames, Iowa (2006–2010).

- 78 original features (structural, quality, location, condition)
- 261 features after one-hot encoding
- Target variable: `SalePrice`

Top correlations with sale price:

```
Overall Qual    0.791    Garage Cars    0.707
Gr Liv Area     0.791    Total Bsmt SF  0.641
Year Built      0.622    Full Bath      0.561
```

-----

## Author

**Mehran Mushtaq**
B.Tech Computer Science & Engineering · IUST 2026
Machine Learning Project

-----

## License

Educational use. The Ames Housing Dataset is publicly available via
[Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
