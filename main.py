import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

housing_data =  pd.read_csv("housing.csv")
housing_data["income_cat"] = pd.cut(housing_data["median_income"], bins=[0,1.5,3,4.5,6,np.inf], labels=[1,2,3,4,5])

sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index, test_index in sss.split(housing_data, housing_data["income_cat"]):
    train_set = housing_data.iloc[train_index].drop("income_cat",axis=1)
    test_set = housing_data.iloc[test_index].drop("income_cat",axis=1)

train_dataset = train_set.copy()


