# libraries
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

# initialise global data
symbol_string = ""
timeframe = ""

while len(symbol_string) <=2:
    symbol_string, timeframe = input("Enter the stock symbol and timeframe: ").upper().split()

stock = yf.Ticker(symbol_string)
stock_df = pd.DataFrame(stock.history(period=timeframe)).reset_index()  # create a df for historical price of the stock

fig = go.Figure(data=[go.Candlestick(x=stock_df['Date'],
                                     open=stock_df['Open'],
                                     high=stock_df['High'],
                                     low=stock_df['Low'],
                                     close=stock_df['Close'])])

fig.update_layout(
    title={
        'text': "Ticker: " + symbol_string,
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    font=dict(
        family="Courier New, monospace",
        size=20,
        color="#7f7f7f"
    ),
    xaxis_rangeslider_visible=False
)

fig.show()
