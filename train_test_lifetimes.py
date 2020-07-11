#!/usr/bin/env python
""" train_test_lifetimes.py
    Plots various graphs depicting the fit of the model:
        plot_calibration_purchases_vs_holdout_purchases
        plot_period_transactions
        plot_cumulative_transactions
        plot_incremental_transactions
    Plots the expected number of purchases on the next day
    & probability alive as a function of recency & frequency
    Trains BG/NBD & Gamma-Gamma model on 80% of data
    Predicts Number of purchases & CLV in next few months
    Evaluates the performance based on Mean Squared Error (MSE) & R-squared (R2) values

    START_DATE (Inclusive): First date to be included
    END_DATE (Non-Inclusive): Last date + 1 to be included
    MONTHS: Number of months to calculate purchases & CLV for (multiplied by 30 for days)
    PLOT: Whether to plot Igraphs
"""

import numpy as np
import pandas as pd
import sqlalchemy as sql
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import *
from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

START_DATE = '2018-06-01'
END_DATE = '2019-12-01'
MONTHS = 3
PLOT = False

sql_engine = sql.create_engine('mysql+mysqldb://root:@localhost/PSP')
df = pd.read_sql_query('select user_id, created_at as datetime, commission as revenue from irctc_booking where pnr_number is not null', sql_engine)
df = df.sort_values(by=['datetime', 'user_id', 'revenue'])
df = df[df['datetime'] >= pd.Timestamp(START_DATE)]
df = df[df['datetime'] < pd.Timestamp(END_DATE) + pd.DateOffset(MONTHS * 30)]
df.loc[df['revenue'] == 0, 'revenue'] = 0.01
df = df.reset_index()

if PLOT:
    data = calibration_and_holdout_data(df, 'user_id', 'datetime', calibration_period_end=END_DATE, monetary_value_col='revenue')
    bgf = BetaGeoFitter()
    bgf.fit(data['frequency_cal'], data['recency_cal'], data['T_cal'])
    plot_calibration_purchases_vs_holdout_purchases(bgf, data)
    plt.savefig('plot_calibration_purchases_vs_holdout_purchases.svg')
    plt.close()

    data = summary_data_from_transaction_data(df, 'user_id', 'datetime', observation_period_end=END_DATE, monetary_value_col='revenue')
    bgf = BetaGeoFitter()
    bgf.fit(data['frequency'], data['recency'], data['T'])

    days = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days
    plot_cumulative_transactions(bgf, df, 'datetime', 'user_id', days + MONTHS*30, days)
    plt.savefig('plot_cumulative_transactions.svg')
    plt.close()
    plot_incremental_transactions(bgf, df, 'datetime', 'user_id', days + MONTHS*30, days)
    plt.savefig('plot_incremental_transactions.svg')
    plt.close()
    plot_period_transactions(bgf)
    plt.savefig('plot_period_transactions.svg')
    plt.close()

    plot_frequency_recency_matrix(bgf)
    plt.savefig('plot_frequency_recency_matrix.svg')
    plt.close()
    plot_probability_alive_matrix(bgf)
    plt.savefig('plot_probability_alive_matrix.svg')
    plt.close()
    # plot_expected_repeat_purchases(bgf)
    # plt.savefig('plot_expected_repeat_purchases.svg')
    # plt.close()
    # plot_history_alive(bgf, MONTHS*30, df, 'datetime')
    # plt.savefig('plot_history_alive.svg')
    # plt.close()

data = summary_data_from_transaction_data(df, 'user_id', 'datetime', observation_period_end=END_DATE, monetary_value_col='revenue')
data_train, data_test = train_test_split(data, train_size=0.8, random_state=0)
bgf = BetaGeoFitter()
bgf.fit(data_train['frequency'], data_train['recency'], data_train['T'])

data_ret = data_train[data_train['frequency'] > 0]
ggf = GammaGammaFitter()
ggf.fit(data_ret['frequency'], data_ret['monetary_value'])

data_test['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(MONTHS * 30, data_test['frequency'], data_test['recency'], data_test['T'])
data_test['predicted_clv'] = ggf.customer_lifetime_value(bgf, data_test['frequency'], data_test['recency'], data_test['T'], data_test['monetary_value'], time=MONTHS, discount_rate=0)

y = df[df['datetime'] > pd.Timestamp(END_DATE)].groupby('user_id')['revenue']
data_test['actual_purchases'] = y.agg('count').reindex(data_test.index).fillna(0)
data_test['actual_clv'] = y.agg('sum').reindex(data_test.index).fillna(0)

print('BetaGeoFitter - Number of purchases prediction:')
data_test.sort_values(by='predicted_purchases').tail(25)[['frequency', 'recency', 'T', 'predicted_purchases', 'actual_purchases']]
print('MSE: ', mean_squared_error(data_test['actual_purchases'], data_test['predicted_purchases']))
print('R2: ', r2_score(data_test['actual_purchases'], data_test['predicted_purchases']))

print('BetaGeoFitter + GammaGammaFitter - Customer lifetime value prediction:')
data_test.sort_values(by='predicted_clv').tail(25)[['frequency', 'recency', 'T', 'monetary_value', 'predicted_clv', 'actual_clv']]
print('MSE: ', mean_squared_error(data_test['actual_clv'], data_test['predicted_clv']))
print('R2: ', r2_score(data_test['actual_clv'], data_test['predicted_clv']))
