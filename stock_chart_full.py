# libraries
import pandas as pd
from plotly import graph_objects as go
import yfinance as yf
import numpy as np

# initialise global data
symbol_string = ""
timeframe = ""

INCREASING_COLOR = '#98FB98'
DECREASING_COLOR = '#FF4500'

def movingaverage(interval, window_size=10):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def bbands(price, window_size=10, num_of_std=5):
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (num_of_std*rolling_std)
    lower_band = rolling_mean - (num_of_std*rolling_std)
    return rolling_mean, upper_band, lower_band

while len(symbol_string) <=2:
    symbol_string, timeframe = input("Enter the stock symbol and timeframe: ").upper().split()

stock = yf.Ticker(symbol_string)
stock_df = pd.DataFrame(stock.history(period=timeframe)).reset_index()  # create a df for historical price of the stock

# stock data in candlestick
data = [dict(
    type='candlestick',
    open=stock_df.Open,
    high=stock_df.High,
    low=stock_df.Low,
    close=stock_df.Close,
    x=stock_df.Date,
    yaxis='y2',
    name=symbol_string
)]

# layout object
layout = dict(
    plot_bgcolor='rgb(250, 250, 250)',
    xaxis=dict(
        rangeselector=dict(
            visible=True,
            x=0, y=0.9,
            bgcolor='rgba(150, 200, 250, 0.4)',
            font=dict(size=13),
            buttons=list([
                dict(count=1,
                     label='reset',
                     step='all'),
                dict(count=1,
                     label='1yr',
                     step='year',
                     stepmode='backward'),
                dict(count=5,
                     label='5 mo',
                     step='month',
                     stepmode='backward'),
                dict(count=3,
                     label='3 mo',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                     label='1 mo',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        )
    ),
    yaxis=dict(domain=[0, 0.2], showticklabels=False),
    yaxis2=dict(domain=[0.2, 0.8]),
    legend=dict(orientation='h', y=0.9, x=0.3, yanchor='bottom'),
    margin=dict(t=40, b=40, r=40, l=40)
)

fig = dict(data=data, layout=layout)

mv_y = movingaverage(stock_df.Close)
mv_x = list(stock_df.Date)

# remove the ends, due to nan
#mv_x = mv_x[5:-5]
mv_y = mv_y[5:-5]

fig['data'].append(dict(
    x=mv_x, y=mv_y,
    type='scatter',
    mode='lines',
    line=dict(width=1),
    marker=dict(color='#696969'),
    yaxis='y2',
    name='Moving Average'
))

# set volume bar chart colours
colours = []

for i in range(len(stock_df.Close)):
    if i != 0:
        if stock_df.Close[i] > stock_df.Close[i-1]:
            colours.append(INCREASING_COLOR)
        else:
            colours.append(DECREASING_COLOR)
    else:
        colours.append(DECREASING_COLOR)

# add volume bar chart
fig['data'].append(dict(
  x=stock_df.Date, y=stock_df.Volume,
    marker=dict(color=colours),
    type='bar', yaxis='y', name='Volume'))

# bollinger bands data
bb_avg, bb_upper, bb_lower = bbands(stock_df.Close)

fig['data'].append(dict(
    x=stock_df.Date, y=bb_upper,
    type='scatter', yaxis='y2',
    line=dict(width=1),
    marker=dict(color='#ccc'), hoverinfo='none',
    legendgroup='Bollinger Bands', name='Bollinger Bands'
))

fig['data'].append(dict(
    x=stock_df.Date, y=bb_lower,
    type='scatter', yaxis='y2',
    line=dict(width=1),
    marker=dict(color='#ccc'), hoverinfo='none',
    legendgroup='Bollinger Bands', showlegend=False
))

# plot
f1 = go.Figure(fig)
f1.show()