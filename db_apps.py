#!/usr/bin/env python
""" db_apps.py

    Reads packages from user_app & connects it to user_id via user_device_mapping
    Sets app = True if the app was installed at any point on any of the user's devices
    Counts the number of competior apps installed
    Prints active user & conversion counts by app & number of apps installed

    PACKAGES: Apps to be audited
"""

import numpy as np
import pandas as pd
import sqlalchemy as sql

PACKAGES = {
            # 'Cleartrip': 'com.cleartrip.android',
            'ConfirmTkT': 'com.confirmtkt.lite',
            # 'GoIbibo': 'com.goibibo',
            'IRCTC Rail Connect': 'cris.org.in.prs.ima',
            'Ixigo': 'com.ixigo.train.ixitrain',
            # 'Make My Trip': 'com.makemytrip',
            'RailYatri': 'com.railyatri.in.mobile',
            # 'Tiket.com': 'com.tiket.gits',
            # 'Where is my Train': 'com.mvision.easytrain',
            # 'Yatra': 'com.yatra.base',
            }

sql_engine = sql.create_engine('mysql+mysqldb://root:@localhost/PSP')

df = pd.read_sql_query('select user_id, device_id from user_device_mapping where user_id is not null', sql_engine)
for name, package in PACKAGES.items():
    p = pd.read_sql_query('select device_id from user_app where package_name = "' + package + '"', sql_engine)
    t = pd.merge(df, p.drop_duplicates(), how='left', on='device_id', indicator=True)
    df[name] = np.where(t._merge == 'both', True, False)

data = pd.DataFrame()
data['user_id'] = df.user_id.unique()
data = data.set_index('user_id')
for name in PACKAGES:
    data[name] = df.groupby(by='user_id')[name].any()

data['Competitors'] = 0
for name in PACKAGES:
    data['Competitors'] += data[name]

data = data.sort_index()
data.to_csv('db_apps.csv')

df = pd.read_sql_query('select id, user_id from irctc_booking where pnr_number is not null', sql_engine)
df2 = pd.read_sql_query('select id, user_id from irctc_booking', sql_engine)

df_c = df['id']
df2_c = df2[['id', 'user_id']]
data_c = pd.merge(df2_c, df_c, on='id', how='left', indicator=True)
data_c['Conversion'] = data_c['_merge'] == 'both'
data_c = pd.merge(data_c, data, on='user_id', how='inner')
data_c.groupby('Competitors')['Conversion'].value_counts().unstack()

data_c['Conversion'].value_counts()
for name in PACKAGES:
    print(name + ': ')
    data_c[data_c[name]]['Conversion'].value_counts()

df_a = pd.DataFrame(df['user_id'].unique(), columns=['user_id'])
df2_a = pd.DataFrame(df2['user_id'].unique(), columns=['user_id'])
data_a = pd.merge(df2_a, df_a, on='user_id', how='left', indicator=True)
data_a['Active'] = data_a['_merge'] == 'both'
data_a = pd.merge(data_a, data, on='user_id', how='inner')
data_a.groupby('Competitors')['Active'].value_counts().unstack()

data_a['Active'].value_counts()
for name in PACKAGES:
    print(name + ': ')
    data_a[data_a[name]]['Active'].value_counts()
