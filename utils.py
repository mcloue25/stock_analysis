import os 
import sys
import json
import math

import numpy as np
import pandas as pd
import yfinance as yf 
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from datetime import date
from itertools import cycle
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def linear_regression(date_list, close_list):

    x = np.array([i for i in range(len(date_list))]).reshape(-1, 1)
    y = np.array(close_list).reshape(-1, 1)

    print("len X", len(x))
    print("len Y", len(y))
    
    # x.reshape(len(x), 1)
    # y.reshape(len(y), 1)

    print("X:", x[0:10])
    print("Y:", y[0:10])

    linreg = LinearRegression().fit(x, y)

    lr_score = linreg.score(x, y)
    print("Score:", lr_score)

    plt.plot(y, color = 'magenta')
    # plt.plot(x, color = 'magenta')
    # plt.plot(x_new, color = 'green')
    plt.show()
    plt.close()


def import_json(json_path):
    '''Loads data from a JSON file into a readable JSON object.
    Args: 
        json_path (JSON): A JSON file containing key point information.
    
    Returns:
        jump_height (Int): The height the user has jumped in centimeters.
    '''
    with open(json_path, "r") as f:
        json_data = json.load(f)

    return json_data

def lowercase_cols(df):
    cols = [i.lower() for i in df.columns]
    df.columns = cols

    return df


def plot_kp_over_time(data, uid):
    ''' Used to plot a dynamic number of signal on top of one another

    Args:
        data (Dict) : A dictionary where each key is a string with the name of the signal & value is the signal in a list
        uid (String) : A string containing the users ID for reference
    '''
    colors = cycle(["blue", "magenta", "red", "green", "fuchsia", "gray", "lime", "maroon", "navy", "olive", "purple", "silver", "teal", "yellow", "aqua", "black"])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle(uid, fontsize=16)

    for key, val in data.items():
        ax.plot(val, label= key, color=next(colors))

    ax.set_xlabel("Frame Index", fontsize=15)
    ax.set_ylabel("Signal vals", fontsize=18)

    ax.legend(loc="best")

    ax.margins(0.1)
    fig.tight_layout()
    plt.show()


def get_historical_data(share_name):
    share = yf.Ticker(share_name)
    historical_data = share.history(period="max")
    share_info = share.info

    data = []
    for row in historical_data.iterrows():
        row_data = row[1]

        data.append({
            "date": str(row[0]),
            "close": row_data["Close"],
            "volume": row_data["Volume"],
            "open": row_data["Open"],
            "low": row_data["Low"],
            "high": row_data["High"]
        })

    return share_info, data

def get_current_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    return todays_data['Close'][0]


def import_json(json_path):
    '''Loads data from a JSON file into a readable JSON object.
    Args: 
        json_path (JSON): A JSON file containing key point information.
    
    Returns:
        jump_height (Int): The height the user has jumped in centimeters.
    '''
    with open(json_path, "r") as f:
        json_data = json.load(f)

    return json_data


# share_info, historical_data = get_historical_data(sys.argv[1])

def display_graph(dists):

    # Colouring lines 
    plt.plot(dists, color = 'magenta')
    # plt.plot(dists2, color = 'green')

    # #plt.savefig(save_path + "/{}.png".format(ref_code))
    # green = mpatches.Patch(color='green', label='leg angles')
    # magenta = mpatches.Patch(color='magenta', label='Spine angles')
    # plt.legend(handles=[green, magenta], bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0)
    plt.show()
    plt.close()


def create_csv(df, path, name):
    print("Creating ", name, ".csv")
    file_path = path + name 
    print("HERE", file_path)
    df.to_csv(file_path)
    return


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    return


def utils_main():

    data = import_json("data/AMD.json")

    # historical_data = data["historicalData"]

    # # display_graph(close_list)
    # stock_df = pd.DataFrame(columns = ["date", "close_price"])
    # stock_df["date"] = date_list
    # stock_df["close_price"] = close_list

    # path = "data/csv/"
    # create_folder(path)
    # create_csv(stock_df, path, "amd.csv")

    stock_df = pd.read_csv("data/csv/amp.csv", delimiter=',', sep=r', ') 

    print(stock_df)

    linear_regression(date_list, close_list)

if __name__ == "__main__":
    utils_main()