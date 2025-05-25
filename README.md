# 🍷 Wine Price Prediction with XGBoost in R

This project demonstrates an end-to-end data science pipeline in **R**, focused on predicting wine prices using features like wine type, rating, region, and more. The analysis includes statistical testing, feature engineering, and a powerful machine learning model: **XGBoost**.

---

## 📌 Project Overview

The goal is to build a regression model that accurately predicts the price of wine based on both **numerical and categorical** attributes. The workflow includes:

- Exploratory Data Analysis (EDA)
- ANOVA and Tukey’s post-hoc test
- Data transformation and encoding
- Feature scaling
- Model training and evaluation with **XGBoost**
- Hyperparameter tuning with `caret`

---

## 📁 Dataset Overview

The dataset `df_wines.csv` contains information such as:

| Column            | Description                      |
|-------------------|----------------------------------|
| `Name`            | Name of the wine                 |
| `Country`         | Country of origin                |
| `Region`          | Wine region                      |
| `Winery`          | Winery name                      |
| `wine_type`       | Type (red, white, rosé, sparkling) |
| `Rating`          | User rating (numeric)            |
| `NumberOfRatings` | Number of user reviews           |
| `Price`           | Target variable (price in euros) |
| `Year1`           | Vintage year                     |

---

## 🔍 Analysis Steps

### 1. 📊 EDA and Preprocessing
- Checked missing values (none found)
- Histogram of prices showed skew → log-transform applied
- Boxplots and ANOVA revealed significant price differences between wine types
- Tukey's test identified key differences (e.g., rosé wines are the cheapest)

### 2. 📈 Feature Engineering
- One-hot encoding of categorical variables: `Name`, `Country`, `Region`, `Winery`, `wine_type`
- Feature scaling applied to numerical features: `Rating`, `NumberOfRatings`, `Price`, `Year1`

### 3. 🤖 Model Training
- Data split: 80% training, 20% test
- Model: **XGBoost** (`reg:squarederror`)
- Evaluation metrics:
  - **R²**: 0.9997 (99.97% variance explained)
  - **RMSE**: 0.018 — very low error

### 4. 🔧 Hyperparameter Tuning
- Used `caret::train()` for random search across:
  - `max_depth`, `eta`, `nrounds`, `gamma`, `subsample`, `colsample_bytree`, `min_child_weight`
- 5-fold cross-validation
- Best parameters printed, though full tuning was **limited by long execution time**

---

## ✅ Conclusions

- The model performed **exceptionally well** on the test set.
- Log transformation and categorical encoding improved results.
- Wines of type **tinto** and **espumoso** are typically more expensive.
- Future work could include additional features (e.g., grape variety), advanced tuning (Bayesian optimization), and outlier treatment.

---

## 🧰 Tools & Libraries

- R 4.x
- `xgboost`
- `caret`
- `dplyr`
- `ggplot2` (optional for visuals)
- `TukeyHSD`, `aov` for statistical tests

---
