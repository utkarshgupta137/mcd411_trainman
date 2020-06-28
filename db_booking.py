import numpy as np
import pandas as pd
import sqlalchemy as sql

START_DATE = '2018-06-01'                               # Inclusive
END_DATE = '2019-12-01'                                 # Non-inclusive
MONTHS = 3                                              # CLV Length (Multiplied by 30)

sql_engine = sql.create_engine('mysql+mysqldb://root:@localhost/PSP')
df = pd.read_sql_query('select user_id, created_at as datetime, commission as revenue from irctc_booking where pnr_number is not null', sql_engine)
df = df.sort_values(by=['datetime', 'user_id', 'revenue'])
df = df[df['datetime'] >= pd.Timestamp(START_DATE)]
df = df[df['datetime'] < pd.Timestamp(END_DATE) + pd.DateOffset(MONTHS * 30)]
df['date'] = df['datetime'].dt.date

data = df[df['date'] < pd.Timestamp(END_DATE)].groupby('user_id')['date'].agg(['min', 'max'])
data['frequency'] = df[df['date'] < pd.Timestamp(END_DATE)].groupby('user_id')['date'].agg(lambda x: x.nunique())
data['recency'] = ((data['max'] - data['min']) / np.timedelta64(1, 'D')).astype(np.int64)
data['T'] = ((pd.Timestamp(END_DATE).date() - data['min']) / np.timedelta64(1, 'D')).astype(np.int64)
data['time_between'] = data['recency'] / data['frequency']
data['revenue'] = df[df['date'] < pd.Timestamp(END_DATE)].groupby('user_id')['revenue'].agg(['sum'])
data['avg_basket_value'] = data['revenue'] / data['frequency']
data['target_revenue'] = df[df['date'] >= pd.Timestamp(END_DATE)].groupby('user_id')['revenue'].agg(['sum']).reindex(data.index).fillna(0)

data = data.drop('min', axis=1).drop('max', axis=1)
data.to_csv('db_booking.csv')
