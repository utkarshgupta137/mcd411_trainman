#!/usr/bin/env python
""" train_tpot.py
    Trains the TPOT model on 80% of the data
    Saves the model as a pipeline
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor

data = pd.read_csv('db_booking.csv')
X = data[['frequency', 'T', 'recency', 'time_between', 'revenue', 'avg_basket_value']]
y = data[['target_revenue']].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

tpot = TPOTRegressor(n_jobs=-1, random_state=0, early_stop=5, verbosity=2)
tpot.fit(X_train, y_train)
tpot.score(X_test, y_test)
tpot.export('tpot_pipeline.py')
