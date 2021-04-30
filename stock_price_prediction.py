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

df = get_stock_price("TLS.AX", "2013-01-01", "2019-12-31")
df = df.set_index("Date")


# Visualise the closing price history
def plot_close(series):
    plt.figure(figsize=(16, 8))
    plt.title('Close Price History')
    plt.plot(series)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price', fontsize=18)
    plt.show()

# Create a new dataframe with only the Close column
data = df.filter(['Close'])
# Convert the dataframe to numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.8)

