import json
import os
import time

import docker
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

client = docker.from_env()
gaml = client.containers.run('gcr.io/cloud-automl-tables-public/model_server', ports={8080: 8080}, volumes={os.getcwd() + '/gaml': {'bind': '/models/default/0000001', 'mode': 'rw'}}, detach=True)
time.sleep(10)

data = pd.read_csv('db_booking.csv')
X = data[['frequency', 'T', 'recency', 'time_between', 'revenue', 'avg_basket_value']]
y = data[['target_revenue']].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

X_test_json = '{"instances":' + X_test.to_json(orient='records') + '}'
r = requests.post('http://localhost:8080/predict', json=json.loads(X_test_json))
y_pred = pd.DataFrame(data=r.json())
print('MSE: ', mean_squared_error(y_test, y_pred))
print('R2: ', r2_score(y_test, y_pred))

gaml.stop()
