import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Load dataset
housing = pd.read_csv("housing.csv")

# 2. Stratified split on income category
housing['income_cat'] = pd.cut(housing["median_income"], 
                               bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

# 3. Separate labels and features
housing = strat_train_set.copy()
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# 4. Define numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# 5. Define pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# 6. Prepare the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

# 7. Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
lin_rmse = np.sqrt(mean_squared_error(housing_labels, lin_preds))
print("Linear Regression RMSE on training set:", lin_rmse)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print("\nLinear Regression CV Results:")
print(pd.Series(lin_rmse_scores).describe())

# 8. Decision Tree Regressor
dec_reg = DecisionTreeRegressor(random_state=42)
dec_reg.fit(housing_prepared, housing_labels)
dec_scores = cross_val_score(dec_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
dec_rmse_scores = np.sqrt(-dec_scores)
print("\nDecision Tree CV Results:")
print(pd.Series(dec_rmse_scores).describe())

# 9. Random Forest Regressor
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(housing_prepared, housing_labels)
rf_preds = rf_reg.predict(housing_prepared)
rf_rmse = np.sqrt(mean_squared_error(housing_labels, rf_preds))
print("Random Forest RMSE on training set:", rf_rmse)

rf_scores = cross_val_score(rf_reg, housing_prepared, housing_labels,
                            scoring="neg_mean_squared_error", cv=10)
rf_rmse_scores = np.sqrt(-rf_scores)
print("\nRandom Forest CV Results:")
print(pd.Series(rf_rmse_scores).describe())
