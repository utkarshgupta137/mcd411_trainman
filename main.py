import numpy as np
import pandas as pd
import seaborn as sns
import sqlalchemy as sql
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

START_DATE = '2018-06-01'                               # Inclusive
END_DATE = '2020-03-16'                                 # Non-inclusive
PLOT = False

pd.set_option('display.float_format', lambda x: '%.5f' % x)
sns.set(rc={'figure.figsize':(6.27*1.5,4*1.5)})
sns.set_palette(['#3366cc', '#dc3912', '#ff9900'])

def order_cluster(cluster_field_name, target_field_name, df, ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df, df_new[[cluster_field_name, 'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name], axis=1)
    df_final = df_final.rename(columns={'index': cluster_field_name})
    return df_final

sql_engine = sql.create_engine('mysql+mysqldb://root:@localhost/PSP')
df = pd.read_sql_query('select user_id, created_at as datetime, commission as revenue from irctc_booking where pnr_number is not null', sql_engine)
df = df.sort_values(by=['datetime', 'user_id', 'revenue'])
df = df[df['datetime'] >= pd.Timestamp(START_DATE)]
df = df[df['datetime'] < pd.Timestamp(END_DATE)]
df['date'] = df['datetime'].dt.date

data = df.groupby('user_id')['date'].agg(['max'])
data['Recency'] = ((pd.Timestamp(END_DATE).date() - data['max']) / np.timedelta64(1, 'D')).astype(np.int64)
data['Frequency'] = df.groupby('user_id')['date'].agg(lambda x: x.nunique())
data['Revenue'] = df.groupby('user_id')['revenue'].agg(['sum'])
data = data.drop('max', axis=1)
data = data.reset_index()

kmeans = KMeans(n_clusters=5, n_init=100, max_iter=1000, random_state=0)

kmeans.fit(data[['Recency']])
data['RecencyCluster'] = kmeans.predict(data[['Recency']])
data = order_cluster('RecencyCluster', 'Recency', data, False)
data.groupby('RecencyCluster')['Recency'].describe()

kmeans.fit(data[['Frequency']])
data['FrequencyCluster'] = kmeans.predict(data[['Frequency']])
data = order_cluster('FrequencyCluster', 'Frequency', data, True)
data.groupby('FrequencyCluster')['Frequency'].describe()

kmeans.fit(data[['Revenue']])
data['RevenueCluster'] = kmeans.predict(data[['Revenue']])
data = order_cluster('RevenueCluster', 'Revenue', data, True)
data.groupby('RevenueCluster')['Revenue'].describe()

data['Score'] = data['FrequencyCluster'] + data['RecencyCluster'] + data['RevenueCluster']
data.groupby('Score')[['Frequency', 'Recency', 'Revenue']].agg({'Frequency': ['count', 'mean', 'min', 'max'], 'Recency': ['mean', 'min', 'max'], 'Revenue': ['mean', 'min', 'max', 'sum']})

data['Segment'] = 'Low-Value'
data.loc[data['Score'] > 4, 'Segment'] = 'Mid-Value'
data.loc[data['Score'] > 8, 'Segment'] = 'High-Value'
data.groupby('Segment')[['Frequency', 'Recency', 'Revenue']].agg({'Frequency': ['count', 'mean', 'min', 'max'], 'Recency': ['mean', 'min', 'max'], 'Revenue': ['mean', 'min', 'max', 'sum']})

if PLOT:
    dp = data[(data['Recency'] < 365) & (data['Frequency'] < 125) & (data['Revenue'] < 4000)]
    hue_order = ['Low-Value', 'Mid-Value', 'High-Value']
    sns.scatterplot(dp['Recency'], dp['Frequency'], hue=dp['Segment'], hue_order=hue_order)
    plt.savefig('rfm_rf.png')
    plt.close()
    sns.scatterplot(dp['Recency'], dp['Revenue'], hue=dp['Segment'], hue_order=hue_order)
    plt.savefig('rfm_rm.png')
    plt.close()
    sns.scatterplot(dp['Frequency'], dp['Revenue'], hue=dp['Segment'], hue_order=hue_order)
    plt.savefig('rfm_fm.png')
    plt.close()

user = pd.read_csv('db_user.csv')
user = pd.merge(data, user, how='inner', on='user_id', suffixes=(False, False))
user.groupby('Score')['age'].describe()
user.groupby('Segment')['age'].describe()
user.groupby('Score')['gender'].value_counts(normalize=True).unstack(level=1).fillna(0)
user.groupby('Segment')['gender'].value_counts(normalize=True).unstack(level=1).fillna(0)
user.groupby('Score')['travel_class'].value_counts(normalize=True).unstack(level=0).fillna(0)
user.groupby('Segment')['travel_class'].value_counts(normalize=True).unstack(level=0).fillna(0)
user.groupby('Score')['quota'].value_counts(normalize=True).unstack(level=0).fillna(0)
user.groupby('Segment')['quota'].value_counts(normalize=True).unstack(level=0).fillna(0)
user.groupby('Score')['city'].value_counts().groupby(level=0).nlargest(20).reset_index(level=0, drop=True).unstack(level=0).fillna(0)
user.groupby('Segment')['city'].value_counts().groupby(level=0).nlargest(20).reset_index(level=0, drop=True).unstack(level=0).fillna(0)

churn = pd.read_csv('db_churn.csv')
churn = pd.merge(data, churn, how='inner', on='user_id', suffixes=(False, False))
churn.groupby('Score')[['prob_alive', 'risk']].agg(['count', 'mean', 'min', 'max'])
churn.groupby('Segment')[['prob_alive', 'risk']].agg(['count', 'mean', 'min', 'max'])

apps = pd.read_csv('db_apps.csv')
apps = pd.merge(apps, data, how='inner', on='user_id', suffixes=(False, False))
apps.groupby('Score')['Competitors'].agg(['count', 'mean', 'min', 'max'])
apps.groupby('Segment')['Competitors'].agg(['count', 'mean', 'min', 'max'])
