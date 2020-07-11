#!/usr/bin/env python
""" test_mdb.py
    Loads & evaluates saved mindsdb model based on Mean Squared Error (MSE) & R-squared (R2) values
"""

import mindsdb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('db_booking.csv')
data = data.drop('user_id', axis=1)
train, test = train_test_split(data, train_size=0.8, random_state=0)

mdb = mindsdb.Predictor(name='mdb')
mdb.load('mdb/mdb.zip')

y_pred = [x.explanation['target_revenue']['predicted_value'] for x in mdb.predict(when_data=test)]
print('MSE: ', mean_squared_error(test['target_revenue'], y_pred))
print('R2: ', r2_score(test['target_revenue'], y_pred))
