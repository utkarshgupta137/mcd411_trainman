#!/usr/bin/env python
""" db_churn.py
    Calculates probability alive for each user using the BG/NBD model
    Segments into risk based on calculated probability

    START_DATE (Inclusive): First date to be included
    END_DATE (Non-Inclusive): Last date + 1 to be included
"""

import numpy as np
import pandas as pd
import sqlalchemy as sql
from lifetimes import BetaGeoFitter
from lifetimes.utils import summary_data_from_transaction_data

START_DATE = '2018-06-01'
END_DATE = '2020-03-16'

sql_engine = sql.create_engine('mysql+mysqldb://root:@localhost/PSP')
df = pd.read_sql_query('select user_id, created_at as datetime from irctc_booking where pnr_number is not null', sql_engine)
df = df.sort_values(by=['datetime', 'user_id'])
df = df[df['datetime'] >= pd.Timestamp(START_DATE)]
df = df[df['datetime'] < pd.Timestamp(END_DATE)]
df = df.reset_index()

data = summary_data_from_transaction_data(df, 'user_id', 'datetime')

bgf = BetaGeoFitter()
bgf.fit(data['frequency'], data['recency'], data['T'])

data['frequency'] = data['frequency'] + 1
data['prob_alive'] = bgf.conditional_probability_alive(data['frequency'], data['recency'], data['T'])
data['risk'] = 0
data.loc[data['prob_alive'] <= 0.5, 'risk'] = '1'
data.loc[data['prob_alive'] <= 0.4, 'risk'] = '2'
data.loc[data['prob_alive'] <= 0.3, 'risk'] = '3'
data.loc[data['prob_alive'] <= 0.2, 'risk'] = '4'
data.loc[data['prob_alive'] <= 0.1, 'risk'] = '5'

data.to_csv('db_churn.csv')
