import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('db_booking.csv')
X = data[['frequency', 'T', 'recency', 'time_between', 'revenue', 'avg_basket_value']]
y = data[['target_revenue']].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

ak = tf.keras.models.load_model('ak')

y_pred = ak.predict(X_test)
print('MSE: ', mean_squared_error(y_test, y_pred))
print('R2: ', r2_score(y_test, y_pred))
