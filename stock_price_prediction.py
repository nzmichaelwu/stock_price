# Import libraries
import math, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date, timedelta
import regex as re

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

from global_functions import add_datepart, add_lags

# variables
prev_date = date.today() - timedelta(days=1)
prev_date_formated = prev_date.strftime("%Y-%m-%d")
N = 10 # number of days to lag

# Get stock quote
def get_stock_price(ticker, startdate, enddate):
    stock = yf.Ticker(ticker)
    stock_df = pd.DataFrame(
        stock.history(start=startdate, end=enddate)).reset_index()  # create a df for historical price of the stock
    return stock_df 


# obtain historical data
df = get_stock_price("TLS.AX", "2013-01-01", prev_date_formated)

# add daily_return column
# df['daily_ret'] = 100 * ((df['Close'] / df['Close'].shift(1)) - 1)

# drop columns
df.drop(['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'], axis=1, inplace=True)

# Change all column headings to be lower case, and remove spacing
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

# # quick plot of the stock price
# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x = df['Date'],
#     y = df['Close'],
#     mode = 'lines'
# ))

# fig.write_html('output/stock price.html')


'''
    Features generation
    - this section is to generate some features for the xgboost regression model to predict the stock price
'''

# # add date fields to the dataframe
# add_datepart(df, 'Date', drop=False)
# df.drop('Elapsed', axis=1, inplace=True)
# df.rename(columns=str.lower, inplace=True)

# add lags up to N number of days to use as features
df_lags = add_lags(df, N, ['close'])

'''
    EDA
'''

# compute correlation
features = [
    'close'
]
for n in range(N, 0, -1):
    features.append("close_lag_"+str(n))

corr_matrix = df_lags[features].corr()
corr_matrix['close'].sort_values(ascending=False) # based on this correlation matrix, we are going to use lags as features, rather than the date features

'''
    Shift label column and drop invalid samples
'''
df_lags['close'] = df_lags['close'].shift(-1)

df_lags = df_lags.loc[10:] # because of close_lag_10
df_lags = df_lags[:-1] # because of shifting close price

df_lags.index = range(len(df_lags))

'''
    train, validation, test split
    70% on train, 15% on validation, 15% on test
'''
test_size = 0.15
val_size = 0.15

test_split_idx = int(df_lags.shape[0] * (1-test_size))
val_split_idx = int(df_lags.shape[0] * (1-(val_size+test_size)))

df_train = df_lags.loc[:val_split_idx].copy()
df_val = df_lags.loc[val_split_idx+1:test_split_idx].copy()
df_test = df_lags.loc[test_split_idx+1:].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_train.date, y=df_train.close, name='Training'))
fig.add_trace(go.Scatter(x=df_val.date, y=df_val.close, name='Validation'))
fig.add_trace(go.Scatter(x=df_test.date,  y=df_test.close,  name='Test'))
fig.show()

# drop unnecessary columns
drop_cols = ['date', 'order_day']

df_train = df_train.drop(drop_cols, axis=1)
df_val = df_val.drop(drop_cols, axis=1)
df_test  = df_test.drop(drop_cols, axis=1)

# split into features and labels
X_train = df_train.drop(['close'], axis=1)
y_train = df_train['close']

X_val = df_val.drop(['close'], axis=1)
y_val = df_val['close']

X_test = df_test.drop(['close'], axis=1)
y_test = df_test['close']

'''
    model build
'''

parameters = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.001, 0.01, 0.03],
    'max_depth': [3, 5, 8],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.5, 0.65, 0.8],
    'colsample_bytree': [0.5, 0.8, 0.9],
    'random_state': [1234]
}

eval_set = [(X_train, y_train), (X_val, y_val)]
model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose=False)
clf = GridSearchCV(model, parameters)

start_time = time.time()
clf.fit(X_train, y_train)
print(f'hyperparameter tuning took {time.time() - start_time} seconds.')
print(f'Best params: {clf.best_params_}')
print(f'Best validation score = {clf.best_score_}')

'''
clf.best_params_
{'colsample_bytree': 0.9,
 'learning_rate': 0.03,
 'max_depth': 8,
 'min_child_weight': 3,
 'n_estimators': 200,
 'random_state': 1234,
 'subsample': 0.65}
'''

model = xgb.XGBRegressor(colsample_bytree=0.9,
                        learning_rate=0.03,
                        max_depth=8,
                        min_child_weight=3,
                        n_estimators=200,
                        subsample=0.65
                    )

model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

'''
    predictions
'''
y_pred = model.predict(X_test)

print(f'mean squared error = {mean_squared_error(y_test, y_pred)}')

predicted_prices = df_lags.loc[test_split_idx+1:][['date', 'close']].copy() # use df_lags here because we drop the first 10 rows above due to lags
predicted_prices['pred_close'] = y_pred

fig = make_subplots(rows=2, cols=1)
fig.add_trace(go.Scatter(x=df.date, y=df.close, # use the original df here
                         name='Truth',
                         marker_color='LightSkyBlue'), row=1, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.date,
                         y=predicted_prices.pred_close,
                         name='Prediction',
                         marker_color='MediumPurple'), row=1, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.date,
                         y=y_test,
                         name='Truth',
                         marker_color='LightSkyBlue',
                         showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=predicted_prices.date,
                         y=y_pred,
                         name='Prediction',
                         marker_color='MediumPurple',
                         showlegend=False), row=2, col=1)

fig.show()