# Import libraries
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import yfinance as yf


# Get stock quote
def get_stock_price(ticker, startdate, enddate):
    stock = yf.Ticker(ticker)
    stock_df = pd.DataFrame(
        stock.history(start=startdate, end=enddate)).reset_index()  # create a df for historical price of the stock
    return stock_df 

df = get_stock_price("TLS.AX", "2013-01-01", "2020-12-31")


train_df = df.iloc[:1620,4:5].values
test_df = df.iloc[1620:,4:5].values