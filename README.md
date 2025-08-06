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

## 📈 Results Summary

| Model                  | Mean RMSE | Std Dev |
|------------------------|-----------|---------|
| Linear Regression      | ~x,xxx.xx | xxx.xx  |
| Decision Tree Regressor| ~x,xxx.xx | xxx.xx  |
| Random Forest Regressor| ~x,xxx.xx | xxx.xx  |

> 🏆 **Random Forest** achieved the best cross-validation performance.

---

## 🧠 Key Learnings

- Importance of stratified sampling when data is imbalanced
- How pipelines simplify preprocessing and keep code clean
- Why decision trees overfit without cross-validation
- Random Forest provides better generalization for structured data

---

## 📌 Future Work

- Perform hyperparameter tuning using GridSearchCV
- Add feature importance and SHAP explainability
- Visualize model predictions vs. actual values
- Deploy the model as a REST API or interactive dashboard

---

## 📁 File Structure

