import os

import numpy as np
import pandas as pd
import sqlalchemy as sql
from fuzzywuzzy import process

START_DATE = '2018-06-01'                               # Inclusive
END_DATE = '2020-03-16'                                 # Non-inclusive

def get_city(x):
    origin = [cities[v] if v in cities else None for v in x['origin'].mode().to_list()]
    dest = [cities[v] if v in cities else None for v in x['dest'].mode().to_list()]
    common = list(set(origin) & set(dest))
    if len(common) == 1:
        return common[0]
    else:
        return np.nan

def optimize(df):
    floats = df.select_dtypes(include=['float']).fillna(-1)
    col_should_be_int = floats.applymap(float.is_integer).all()
    float_to_int_cols = col_should_be_int[col_should_be_int].index
    df.loc[:, float_to_int_cols] = floats[float_to_int_cols].astype(int)
    floats = df.select_dtypes(include=['float']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    ints = df.select_dtypes(include=['integer']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df

def reduce(df, f):
    pd.DataFrame(columns=['user_id', 'age', 'gender', 'city', 'travel_class', 'quota', 'food_choice', 'berth_choice', 'opted_berth', 'opted_ss_concession', 'bedroll_choice']).to_csv(f, index=False)
    for _, g in df.groupby('user_id'):
        if not pd.isnull(g['dob'].iloc[0]):
            g['age_x'] = (g['updated_at'] - g['dob']) / np.timedelta64(1, 'Y')
            g = g[g['age'] - g['age_x'] < 3]
            g['age'] = g['age_y']
        if len(g.index) > 0:
            if not pd.isnull(g['gender_y'].iloc[0]):
                g = g[g['gender'] == g['gender_y']]
            if len(g.index) > 1:
                if not pd.isnull(g['name_y'].iloc[0]):
                    g = g.loc[[v[2] for v in process.extractBests(g['name_y'].iloc[0], g['name'], score_cutoff=60, limit=10)]]
                if len(g.index) > 1:
                    g = g[g['updated_at'] == g['updated_at'].max()]
                    if len(g.index) > 1:
                        g = g[g['created_at'] == g['created_at'].max()]
                        if len(g.index) > 1:
                            g = g.iloc[0]
        if len(g.index) > 0:
            g[['user_id', 'age', 'gender', 'city', 'travel_class', 'quota', 'food_choice', 'berth_choice', 'opted_berth', 'opted_ss_concession', 'bedroll_choice']].to_csv(f, header=False, index=False)

sql_engine = sql.create_engine('mysql+mysqldb://root:@localhost/PSP')

passenger = pd.read_sql_query('select user_id, created_at, updated_at, name, age, gender, food_choice, berth_choice, opted_berth, opted_ss_concession, bedroll_choice from irctc_passenger where user_id in (select distinct(user_id) from irctc_booking where pnr_number is not null)', sql_engine)
passenger = passenger[passenger['created_at'] < pd.Timestamp(END_DATE)]
passenger = passenger.sort_values(['user_id', 'updated_at'])
passenger = passenger.drop_duplicates(subset=['user_id', 'name', 'age', 'gender', 'food_choice', 'berth_choice', 'opted_berth', 'opted_ss_concession', 'bedroll_choice'], keep='last')

user = pd.read_sql_query('select id as user_id, name, dob, gender from user', sql_engine)
user = user.set_index('user_id')
user['dob'] = pd.to_datetime(user['dob'], errors='coerce')
user['age'] = ((pd.Timestamp(END_DATE) - user['dob']) / np.timedelta64(1, 'Y')).apply(np.floor)

booking = pd.read_sql_query('select user_id, origin, dest, travel_class, quota from irctc_booking where pnr_number is not null', sql_engine)
user['travel_class'] = booking.groupby('user_id')['travel_class'].agg(lambda x: x.mode()[0])
user['quota'] = booking.groupby('user_id')['quota'].agg(lambda x: x.mode()[0])

city = pd.read_sql_query('select city.name as city, train_station.code from city join train_station on city.id = train_station.city_id', sql_engine)
cities = city.set_index('code').to_dict()['city']
user['city'] = booking.groupby('user_id')[['origin', 'dest']].apply(get_city)

df = pd.merge(passenger, user, on='user_id', how='left', suffixes=('', '_y'))
df = optimize(df)

if os.path.exists('db_user.csv'):
    os.remove('db_user.csv')

with open('db_user.csv', 'a') as f:
    reduce(df, f)
