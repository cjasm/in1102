# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor
from tmdb_preprocessing import preprocessing
"""
//TO DO
calculate the log of revenue and budget
cross validation and holdout
confiance interval 
"""

dataset = pd.read_csv('../data/train.csv')

X, y = preprocessing(dataset)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

params = {'num_leaves': 30,
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': 5,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2,
         "verbosity": -1}

# Training the dataset
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(X_train, y_train)
tree_regressor = DecisionTreeRegressor(random_state=0)
tree_regressor.fit(X_train, y_train)
forest_regressor = RandomForestRegressor(n_estimators=300, random_state=0)
forest_regressor.fit(X_train, y_train)
lgbm_regressor = LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)
lgbm_regressor.fit(X_train, y_train)


# Predicting a new result
y_pred_linear = linear_regressor.predict(X_test)
y_pred_svr = svr_regressor.predict(X_test)
y_pred_tree = tree_regressor.predict(X_test)
y_pred_forest = forest_regressor.predict(X_test)
y_pred_lgbm = lgbm_regressor.predict(X_test)

# Evaluating
print("Multiple Linear:", np.sqrt(mean_squared_log_error( y_test, y_pred_linear )))
print("SVR:", np.sqrt(mean_squared_log_error( y_test, y_pred_svr )))
print("Tree:", np.sqrt(mean_squared_log_error( y_test, y_pred_tree )))
print("Forest:", np.sqrt(mean_squared_log_error( y_test, y_pred_forest )))
print("LGBM:", np.sqrt(mean_squared_log_error( y_test, y_pred_lgbm )))