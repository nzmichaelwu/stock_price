# libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import requests
import json
from datetime import datetime
from matplotlib import rcParams
import plotly.express as px
import plotly.graph_objects as go

# initialise global data
url = "https://apidojo-yahoo-finance-v1.p.rapidapi.com/market/get-charts"

RAPIDAPI_HOST = "apidojo-yahoo-finance-v1.p.rapidapi.com"
RAPIDAPI_KEY = "fe2e4e9339msh10d3ef8fd79cb50p1e3229jsnc36e0535eb57"

symbol_string = ""
region_string = ""
inputdata = {}

def fetchstockdata(symbol, region):
    querystring = {"comparisons": "%5EGDAXI%2C%5EFCHI", "region": region, "lang": "en", "symbol": symbol, "interval": "1d",
                   "range": "1y"}

    headers = {
        'x-rapidapi-host': RAPIDAPI_HOST,
        'x-rapidapi-key': RAPIDAPI_KEY
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    if(response.status_code == 200):
        return response.text
    else:
        return None

def parsetimestamp(data):
    timestamplist = []

    timestamplist.extend(data["chart"]["result"][0]["timestamp"])   # for open price
    timestamplist.extend(data["chart"]["result"][0]["timestamp"])   # for close price

    calendartime = []

    for ts in timestamplist:
        dt = datetime.fromtimestamp(ts)
        calendartime.append(dt.strftime("%d/%m/%Y"))

    return calendartime

def parsevalues(data):
    valuelist = []
    valuelist.extend(data["chart"]["result"][0]["indicators"]["quote"][0]["open"])
    valuelist.extend(data["chart"]["result"][0]["indicators"]["quote"][0]["close"])

    return valuelist

def attachevents(data):
    eventlist = []
    for i in range(0,len(data["chart"]["result"][0]["timestamp"])):
        eventlist.append("open")

    for i in range(0,len(data["chart"]["result"][0]["timestamp"])):
        eventlist.append("close")

    return eventlist

while len(symbol_string) <=2:
    symbol_string, region_string = input("Enter the stock symbol: ").upper().split()

retdata = fetchstockdata(symbol_string, region_string)
retdata = json.loads(retdata)

if (None != inputdata):
    inputdata["Timestamp"] = parsetimestamp(retdata)
    inputdata["Values"] = parsevalues(retdata)
    inputdata["Events"] = attachevents(retdata)

    df = pd.DataFrame(inputdata)

    print(df)

    sns.set(style="darkgrid")
    rcParams['figure.figsize'] = 13, 5
    rcParams['figure.subplot.bottom'] = 0.2
    ax = sns.lineplot(x="Timestamp", y="Values", hue="Events", dashes=False, markers=True, data=df, sort=False)
    ax.set_title('Symbol: ' + symbol_string)

    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontweight='light',
        fontsize='xx-small'
    )

    plt.show()


    
    

