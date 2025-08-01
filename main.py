import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load and preprocess
housing_data = pd.read_csv("housing.csv")
housing_data["income_cat"] = pd.cut(housing_data["median_income"],
                                    bins=[0,1.5,3,4.5,6,np.inf],
                                    labels=[1,2,3,4,5])

# Stratified split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(housing_data, housing_data["income_cat"]):
    train_set = housing_data.loc[train_index].drop("income_cat", axis=1)
    test_set = housing_data.loc[test_index].drop("income_cat", axis=1)

train_dataset = train_set.copy()

# Separate features and labels
train_labels = train_dataset["median_house_value"]
train_features = train_dataset.drop("median_house_value", axis=1)

# Attributes
cat_attributes = ["ocean_proximity"]
num_attributes = train_features.drop("ocean_proximity", axis=1).columns.tolist()

# Pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Column transformer
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", cat_pipeline, cat_attributes),
])

# Apply transformations
train_prepared = full_pipeline.fit_transform(train_features)
