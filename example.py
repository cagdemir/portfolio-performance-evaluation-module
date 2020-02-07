#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: cagri
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(5)

#------------------------------------------------------------------------------
data_path = './data/'
data_daily = pd.read_excel(data_path+'s&p500_daily.xlsx')

data_daily.index = pd.to_datetime(data_daily.Date)
port_original_daily = data_daily['Price Close'].pct_change().iloc[1:]
port_original_daily.name = 'SP500'

high_daily = port_original_daily.shape[0]
len_port_daily = 252 * 10
n_port = 1000

market_daily = port_original_daily.iloc[-len_port_daily:]


# constructing portfolios

idx_daily = np.random.randint(0,high_daily,(len_port_daily,n_port))

df_port_rets_daily = pd.DataFrame(port_original_daily.values[idx_daily], columns = list(range(n_port)), index = list(range(len_port_daily))) 
df_port_rets_daily.index = market_daily.index

corr_port_daily = df_port_rets_daily.corr().values[np.triu_indices(df_port_rets_daily.shape[1], k=1)]

print(pd.Series(corr_port_daily).describe())
plt.hist(corr_port_daily, bins=1000)

#------------------------------------------------------------------------------

# monthly _test

market_monthly = (market_daily+1).groupby(pd.Grouper(freq='M')).apply(np.prod) - 1

#data_rf_monthly = pd.read_excel('/home/research/Desktop/portfolio_metrics_project/data/M_TB3MS.xls')
#data_rf_monthly = data_rf_monthly.iloc[:,1]

#rf_monthly = data_rf_monthly.iloc[-len_port_monthly:] / 12
#rf_monthly.index = market_monthly.index

rf_monthly=None

# constructing portfolios


df_port_rets_monthly = (df_port_rets_daily+1).groupby(pd.Grouper(freq='M')).apply(np.prod) - 1

corr_port_monthly = df_port_rets_monthly.corr().values[np.triu_indices(df_port_rets_monthly.shape[1], k=1)]

print(pd.Series(corr_port_monthly).describe())
plt.hist(corr_port_monthly, bins=1000)
plt.show()

# calculating metrics

port_metrics_monthly = df_port_rets_monthly.apply(lambda x: pd.Series(performance_metrics(x, market=market_monthly, rf=rf_monthly, freq='M')), axis=0).T

corr_port_metrics_monthly = port_metrics_monthly.corr()
print(corr_port_metrics_monthly)


#------------------------------------------------------------------------------

# weekly _test

market_weekly = (market_daily+1).groupby(pd.Grouper(freq='W')).apply(np.prod) - 1

#data_rf_weekly = pd.read_excel('/home/research/Desktop/portfolio_metrics_project/data/M_TB3MS.xls')
#data_rf_weekly = data_rf_weekly.iloc[:,1]

#rf_weekly = data_rf_weekly.iloc[-len_port_weekly:] / 12
#rf_weekly.index = market_weekly.index

rf_weekly=None

# constructing portfolios


df_port_rets_weekly = (df_port_rets_daily+1).groupby(pd.Grouper(freq='W')).apply(np.prod) - 1

corr_port_weekly = df_port_rets_weekly.corr().values[np.triu_indices(df_port_rets_weekly.shape[1], k=1)]

print(pd.Series(corr_port_weekly).describe())
plt.hist(corr_port_weekly, bins=1000)

# calculating metrics

port_metrics_weekly = df_port_rets_weekly.apply(lambda x: pd.Series(performance_metrics(x, market=market_weekly, rf=rf_weekly, freq='W')), axis=0).T

corr_port_metrics_weekly = port_metrics_weekly.corr()
print(corr_port_metrics_weekly)

#------------------------------------------------------------------------------

# annual _test

market_annual = (market_daily+1).groupby(pd.Grouper(freq='A')).apply(np.prod) - 1

#data_rf_annual = pd.read_excel('/home/research/Desktop/portfolio_metrics_project/data/M_TB3MS.xls')
#data_rf_annual = data_rf_annual.iloc[:,1]

#rf_annual = data_rf_annual.iloc[-len_port_annual:] / 12
#rf_annual.index = market_annual.index

rf_annual=None

# constructing portfolios


df_port_rets_annual = (df_port_rets_daily+1).groupby(pd.Grouper(freq='A')).apply(np.prod) - 1

corr_port_annual = df_port_rets_annual.corr().values[np.triu_indices(df_port_rets_annual.shape[1], k=1)]

print(pd.Series(corr_port_annual).describe())
plt.hist(corr_port_annual, bins=1000)

# calculating metrics

port_metrics_annual = df_port_rets_annual.apply(lambda x: pd.Series(performance_metrics(x, market=market_annual, rf=rf_annual, freq='Y')), axis=0).T

corr_port_metrics_annual = port_metrics_annual.corr()
print(corr_port_metrics_annual)


