#!/usr/bin/env python
""" test_tpot.py
    Fits & evaluates TPOT pipeline based on Mean Squared Error (MSE) & R-squared (R2) values
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, Normalizer
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from xgboost import XGBRegressor

data = pd.read_csv('db_booking.csv')
X = data[['frequency', 'T', 'recency', 'time_between', 'revenue', 'avg_basket_value']]
y = data[['target_revenue']].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

tpot = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    Normalizer(norm='max'),
    MinMaxScaler(),
    MinMaxScaler(),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.25, tol=0.001)),
    XGBRegressor(learning_rate=0.1, max_depth=3, min_child_weight=5, n_estimators=100, nthread=1, objective='reg:squarederror', subsample=0.7)
)
set_param_recursive(tpot.steps, 'random_state', 0)
tpot.fit(X_train, y_train)

results = tpot.predict(X_test)
y_pred = pd.DataFrame(data=results)
print('MSE: ', mean_squared_error(y_test, y_pred))
print('R2: ', r2_score(y_test, y_pred))
