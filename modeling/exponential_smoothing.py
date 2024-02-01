from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import os
import sys
import pandas as pd

path = os.getcwd()
parent_path = os.path.abspath(os.path.join(path, os.pardir))
data_path = str(parent_path) + "/Forecastor/data_processing"

sys.path.append(data_path)


from data_handler import get_data, get_stores, get_families, get_time_series

# single exponential smoothing
def exp_smoothing_forecast(data):
    # create class
    model = SimpleExpSmoothing(data)
    # fit model
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.forecast(6)
    return yhat

df = get_data()
print(df.head())
table = pd.pivot_table(df, values='sales', index=['store_nbr', 'family'], aggfunc="sum")
print(table.head())
df_time_serie = df[(df['family']=="BEVERAGES")&(df['store_nbr']==1)]
df_time_serie = df_time_serie[['date', 'sales']]
df_time_serie = df_time_serie.set_index('date')
print(df_time_serie)
yhat = exp_smoothing_forecast(df_time_serie)
print(yhat)