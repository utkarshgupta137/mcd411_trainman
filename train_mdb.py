#!/usr/bin/env python
""" train_mdb.py
    Trains the mindsdb model on 80% of the data
    Saves the model as mdb.zip
    Note: Run in mindsdb container
"""

import mindsdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('db_booking.csv')
data = data.drop('user_id', axis=1)
train, test = train_test_split(data, train_size=0.8, random_state=0)

mdb = mindsdb.Predictor(name='mdb')
mdb.learn(from_data=train, to_predict='target_revenue')
mdb.export_model()
