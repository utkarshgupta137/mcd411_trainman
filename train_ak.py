import numpy as np
import pandas as pd
from autokeras import StructuredDataRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv('db_booking.csv')
X = data[['frequency', 'T', 'recency', 'time_between', 'revenue', 'avg_basket_value']]
y = data[['target_revenue']].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

ak = StructuredDataRegressor(seed=0)
ak.fit(X_train, y_train, epochs=100)
ak.evaluate(X_test, y_test)
ak.export_model().save('ak')
