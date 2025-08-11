# 🏡 California Housing Price Prediction

This project is focused on building machine learning models to predict **median house values** in California using the California Housing Dataset. The project involves full data preprocessing, feature engineering, model training, and evaluation using cross-validation.

---

## 📂 Dataset

- **Source**: California Housing Dataset (originally from the 1990 U.S. Census)
- **Format**: CSV file (`housing.csv`)
- **Features**:
  - `longitude`, `latitude`
  - `housing_median_age`
  - `total_rooms`, `total_bedrooms`
  - `population`, `households`
  - `median_income`
  - `ocean_proximity` (categorical)
  - `median_house_value` (target)

---

## 🧠 Project Goals

- Perform data cleaning and preprocessing using Scikit-Learn Pipelines
- Split data using **Stratified Sampling** based on income category
- Train and evaluate three different regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Use **cross-validation** to estimate model performance and avoid overfitting

---

## ⚙️ Tools & Libraries

- Python
- Pandas, NumPy
- Scikit-Learn (`Pipeline`, `ColumnTransformer`, `Imputer`, `StandardScaler`, `OneHotEncoder`, `cross_val_score`)
- Matplotlib (optional for visualization)

---

## 🔄 Workflow Overview

1. **Load Data**  
   Load and explore the dataset using `pandas`.

2. **Stratified Split**  
   Create an `income_cat` feature and apply `StratifiedShuffleSplit` to ensure fair distribution.

3. **Preprocessing Pipelines**  
   - Numerical attributes: Median imputation + Standardization
   - Categorical attributes: One-hot encoding

4. **Model Training**  
   Train three models on the processed training data:
   - `LinearRegression`
   - `DecisionTreeRegressor`
   - `RandomForestRegressor`

5. **Model Evaluation**  
   - Evaluate models using **Root Mean Squared Error (RMSE)**
   - Apply **10-fold cross-validation** for robust evaluation

---

---

## 📈 Results Summary

| Model                     | Mean RMSE | Std Dev | Min RMSE | Max RMSE |
|---------------------------|-----------|---------|-----------|-----------|
| Linear Regression         | 69,204    | 2,500   | 65,318    | 73,003    |
| Decision Tree Regressor  | 69,195    | 2,165   | 65,611    | 72,665    |
| Random Forest Regressor  | 49,403    | 2,010   | 46,173    | 52,873    |

> ✅ **Random Forest** outperformed the other models with the lowest RMSE and lowest variance across 10-fold cross-validation.

---

## 🧠 Key Insights

- **Linear Regression** and **Decision Trees** have similar RMSEs (~69,200) with moderate variance.
- **Random Forest Regressor** achieved the best performance with a **much lower average RMSE (~49,400)** and more consistent results, indicating better generalization.
- This highlights the power of ensemble methods like Random Forest for regression problems with complex feature interactions.

---

## 📌 Future Work

- Perform hyperparameter tuning using GridSearchCV
- Add feature importance and SHAP explainability
- Visualize model predictions vs. actual values
- Deploy the model as a REST API or interactive dashboard

