import os
import ta 
import time
import datetime

import numpy as np 
import pandas as pd
import investpy as investpy
# import pandas_ta as ta

from pyfinance import TSeries
from ta import add_all_ta_features

from utils import *


def SortinoRatio(df, T):
    ''' Calculates the Sortino ratio from univariate excess returns.
    Args:
        df ([float]): The dataframe or pandas series of univariate excess returns.
        T ([integer]): The targeted return. 
    '''
    temp = np.minimum(0, df - T)**2
    temp_expectation = np.mean(temp)
    downside_dev = np.sqrt(temp_expectation)
    sortino_ratio = np.mean(df - T) / downside_dev

    return sortino_ratio


def calculate_information_ratio(df, window_length):
    ''' Read in visual results and extract all participant ID's
        https://www.youtube.com/watch?v=E6M89IHpmzc
    '''
    spy_df = pd.read_csv("data/csv/SPY.csv", delimiter=',', sep=r', ') 
    spy_df.set_index(["date"], inplace = True)

    spy_df["daily_returns"] = spy_df['close'] - spy_df['open'] 
    bench_return = spy_df["daily_returns"]
    bench_return_prod = np.prod(spy_df["daily_returns"] + 1) - 1

    # print('bench_return', bench_return)
    # print("---------------")
    # print('bench_return_prod', bench_return_prod)
    df['{}d_information_ratio'.format(window_length)] = df['daily_returns'].rolling(window_length).apply(get_information_ratio, args=(bench_return, bench_return_prod, window_length))

    return df

def get_information_ratio(returns, bench_return, bench_return_prod, window_length):
    ''' Information ration = (stock return - benchmark return) / tracking)
    '''
    dates = returns.index
    bench_return = bench_return.loc[min(dates) : max(dates)]
    df_return_prod = np.prod(returns + 1) - 1

    tracking_error = (returns - bench_return).std() * np.sqrt(window_length)
    information_ratio = (returns - bench_return_prod) / tracking_error

    return information_ratio


def create_df_from_json(json_path):
    ''' Used to extract all neccessary data from the stock JSON files and say them to a DataFrame
    '''
    data = import_json(json_path)
    historical_data = data["historicalData"]

    data_map = {}
    keys_list = list(historical_data[0].keys())
    for col_name in keys_list:
        data_map[col_name] = []

    # Iterate through JSOn and collect data
    df = pd.DataFrame(columns = keys_list)
    for day_results in historical_data:
            # Extract info and save to the appropriate list in data_map
        for key_name in keys_list:
            if key_name == "date":
                data_map[key_name].append(day_results[key_name].split(" ")[0])
            else:
                data_map[key_name].append(day_results[key_name])

        # Populate DF with values
    for col_name, vals in data_map.items():
        df[col_name] = vals

    # Numpy Financial: https://numpy.org/numpy-financial/
    # df["internal_rate_of_return"] = npf.irr(np.array(df["close"]))

    df.columns = [i if i != "price" else i.replace("price", "close") for i in df.columns]
    df.set_index(["date"], inplace = True)
    stock_name = json_path.split("/")[2].replace(".json", ".csv")

    return df


def calculate_inflation_rates(path, name):
    ''' Calculate the CPI multiplier for each row in the CSV 
    Args: 
        path (String) : A string file path to the folder containing the inflation CSV's 
        name (String) : The name of the of the CSV file containin the inflation info
    '''
    inflation_df = pd.read_csv(path + name, delimiter=',', sep=r', ')
    inflation_df["cpiaucns_multiplier"] = inflation_df["cpiaucns"].iloc[-1] / inflation_df["cpiaucns"]

    # Rename column names to lowercase to keep consistency
    col_names = [i.lower() for i in inflation_df.columns]
    inflation_df.columns = col_names

    inflation_df.set_index(["date"], inplace = True)
    create_csv(inflation_df, path, name)

    return inflation_df


def get_inflation_price_adjustments(df, inflation_df):
    ''' Used to calculate the percentage agreement for each row in the agreement_df 
    
    Args: 
        df (DataFrame) : DataFrame containing the stock data with prices to be adjusted for inflation
        inflation_df (DataFrame) : DataFrame containing the inflation data
    '''
    date_vals = list(df.index)
    inf_date_vals = list(inflation_df.index)

    df = pd.merge(df, inflation_df, how='left', on='date', suffixes=('', '_delme'))
    df = df[[c for c in df.columns if not c.endswith('_delme')]]
    df["cpi_adj_price"] = df["close"] * df["cpiaucns_multiplier"]
    df = df.ffill()

    return df


def calc_volume_and_gain_loss_avgs(df):
    ''' Calculates  The 20 & 100 day  volume averages
                    The daily gain/loss
                    The average gain/loss over 7, 14, 50 & 200 day periods
    Args: 
        df (DataFrame) : DataFrame containing the stock data with prices to be adjusted for inflation
    '''
    df["20d_vol_avg"] = df['volume'].rolling(20).mean()
    df["100d_vol_avg"] = df['volume'].rolling(100).mean()

    df['daily_diff'] = df['close'].diff(1)
    df['daily_returns'] = df['close'].pct_change()
    # df['daily_diff_pc_change'] = df['daily_diff'].pct_change()

    df['daily_volume_diff'] = df['volume'].diff(1)
    df['daily_volume_pc_change'] = df['daily_volume_diff'].pct_change()

    # Calculate Avg. Gains/Losses
    df['gain'] = df['daily_diff'].clip(lower=0).round(2)
    df['loss'] = df['daily_diff'].clip(upper=0).abs().round(2)
    
    # Calculate average gain losse for a period of time 
    df = calculate_average_gain_loss(df,7)
    df = calculate_average_gain_loss(df, 14)
    df = calculate_average_gain_loss(df, 50)
    df = calculate_average_gain_loss(df, 200)
    
    return df


def calculate_average_gain_loss(df, window_length):
    ''' Calculates the average gain and loss over a given time period as well as the RS and RSI values
    Args: 
        df (DataFrame) : DataFrame containing the stock data with prices to be adjusted for inflation
        window_length (Integer) : The rolling time period we want to calculate the Shapre ratio for
    '''
    df['{}d_avg_gain'.format(window_length)] = df['gain'].rolling(window_length).mean()
    df['{}d_avg_loss'.format(window_length)] = df['loss'].rolling(window_length).mean()

    # Calculate the percentage gain and loss over the specified time period  -- WASNT USEFUL --
    # df['{}d_avg_gain_pc_change'.format(window_length)] = df['gain'].pct_change(periods=window_length)
    # df['{}d_avg_loss_pc_change'.format(window_length)] = df['loss'].pct_change(periods=window_length)
    
    # Calculate RS & RSI Values
    df['{}d_rs'.format(window_length)] = df['{}d_avg_gain'.format(window_length)] / df['{}d_avg_loss'.format(window_length)]
    df['{}d_rsi'.format(window_length)] = 100 - (100 / (1.0 + df['{}d_rs'.format(window_length)]))

    return df


def calculate_sharpe_ratio(df, window_length):
    ''' Calculates the Sharpe Ratio over a given time period. That is, the average return of the investment ddivided by the standard deviation
    Args:
        df (DataFrame) : DataFrame containing the stock data with prices to be adjusted for inflation
        window_length (Integer) : The rolling time period we want to calculate the Shapre ratio for
    '''
    df['{}d_sharpe_ratio'.format(window_length)] = df['daily_returns'].rolling(window_length).apply(sharpe_ratio)

    return df


def sharpe_ratio(data, risk_free_rate=0.0):
    ''' Sharpe ratio calculation from: https://medium.datadriveninvestor.com/the-sharpe-ratio-with-python-from-scratch-fbb1d5e490b9
    
    Args:
        data (DataFrame) : A rolling window of values over a given timer period
        risk_free_rate (integer) : Assume no trade is risk free

    Returns:
        Sharpe ratio for that given window of time
    '''
    arr = np.array(data.values)
    idx = pd.date_range(start = min(data.index), periods = len(data.values))  # default daily freq.
    ts = TSeries(arr, index=idx)
    sharpe_ratio = ts.sharpe_ratio(ddof=1)

    return sharpe_ratio


def calculate_maximum_drowdown(df):
    ''' Calculates the maximum dropdown over a given period of time

    Args:
        df (DataFrame) : DataFrame containing the stock data with prices to be adjusted for inflation
        window_length (Integer) : The rolling time period we want to calculate the Shapre ratio for
    Returns:
        df (DataFrame) : DataFrame with the Sharpe ratio added for that time period
    '''
    df = maximum_dropdown(df, 7)
    return df


def maximum_dropdown(df, window_length):
    ''' Main function for calculating various volatility metric scores over a rolling range of time periods

    Args:
        df (DataFrame) : DataFrame containing the stock data with prices to be adjusted for inflation

    Returns:
        df (DataFrame) : DataFrame with all newly created columns addded
    '''
    rolling_max = df['close'].rolling(window_length, min_periods=1).max()
    daily_dropdown = df['close'] / rolling_max - 1.0
    Max_Daily_Drawdown = daily_dropdown.rolling(window_length, min_periods=1).min()

    return df


def get_volatility_scores(df):
    ''' Main function for calculating various volatility metric scores over a rolling range of time periods

    Args:
        df (DataFrame) : DataFrame containing the stock data with prices to be adjusted for inflation

    Returns:
        df (DataFrame) : DataFrame with all newly created columns addded
    '''
    # Calculates the historic volatility over a given time period
    df = calculate_historic_volatility(df, 7)
    df = calculate_historic_volatility(df, 14)
    df = calculate_historic_volatility(df, 252)

    # Calculate the Sharpe Ratio over a given time period
    df = calculate_sharpe_ratio(df, 7)
    df = calculate_sharpe_ratio(df, 14)
    df = calculate_sharpe_ratio(df, 50)
    df = calculate_sharpe_ratio(df, 200)

    return df

def calculate_historic_volatility(df, window_length):
    ''' Calculates the volatility of a stock over aa given time period

    '''
    df['log_ret'] = np.log(df.close) - np.log(df.close.shift(1))
    # Compute Volatility using the pandas rolling standard deviation function
    df['{}d_historic_volatility_of_risk-adjusted_return'.format(window_length)] = df['log_ret'].rolling(window=window_length).std() * np.sqrt(window_length)

    return df


def calculate_force_and_comodity_indexes(df):
    ''' Calculate force index & comodity chanel index for varying day ranges
    '''
    calc_force_index(df, 7)
    calc_force_index(df, 14)

    calc_comodity_chanel_index(df, 7)
    calc_comodity_chanel_index(df, 14)

    return df


def calc_force_index(df, window_length):
    ''' Calculate the force index over a given period of time
    ''' 
    df["{}d_force_index".format(window_length)] = df['close'].diff(window_length) * df['volume'] 
    
    return df


def calc_comodity_chanel_index(df, window_length):
    ''' Calculate comodity chanel index over a given period of time
    ''' 
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3 
    df['sma'] = df['TP'].rolling(window_length).mean()
    df['mad'] = df['TP'].rolling(window_length).apply(lambda x: pd.Series(x).mad())
    df['{}d_cci'.format(window_length)] = (df['TP'] - df['sma']) / (0.015 * df['mad']) 

    del df["TP"]
    del df["sma"]
    del df["mad"]
    
    return df


def get_crypto_data():
    ''' Used for downloading historic statistical data for a list of crypto coins
    '''
    crypto_df = pd.read_csv("data/coin_names.csv", delimiter=',', sep=r', ')
    crypto_df.set_index("name", inplace = True)
    time.sleep(10)

    for coin_name in crypto_df.index:
        (start_date, end_date) = get_dates(coin_name)
        sd = start_date.split("-")
        ed = end_date.split("-")
        e = ed[2] + "/" + ed[1] + "/" + ed[0]
        s = sd[2] + "/" + sd[1] + "/" + sd[0]
        
        df = investpy.get_crypto_historical_data(crypto = coin_name,  from_date = s, to_date=e)
        create_csv(df, "data/csv/", coin_name + ".csv")
        time.sleep(10)


def get_dates(coin_name):
    ''' Return the first & most recent occuring dates for a given coin name
    '''
    path = "data/other/csv/" + coin_name.lower() + ".csv"
    df = pd.read_csv(path, delimiter=',', sep=r', ')

    return(min(df["date"]), max(df["date"]))


def generate_features(df):
    ''' Creates features used to assess a stock
    '''
    df = get_inflation_price_adjustments(df, inflation_df)
    df = calc_volume_and_gain_loss_avgs(df)
    df = calculate_maximum_drowdown(df)
    df = calculate_force_and_comodity_indexes(df)
    df = get_volatility_scores(df)

    # Calculate information ratio
    # df = calculate_information_ratio(df, 7)

    return df

def json_to_df(json_path):
    ''' Takes a stock json path & retuns the df
    '''
    df = pd.read_csv(json_path, delimiter=',', sep=r', ')
    df = lowercase_cols(df)
    df.set_index("date", inplace = True)
    df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume")
    return df


def feature_engineering_main():

    # https://github.com/alvarobartt/investpy/issues/467 BLocked by investing.com for multiple calls, use sleep to avoid in future
    # get_crypto_data()

    csv_path = "data/csv/" 
    json_folder_path = "data/csv/"

    # Only needs to be ran if new Inflation CSV is added 
    # calculate_inflation_rates("data/us_inflation_rates/", "CPIAUCNS.csv")
    inflation_df = pd.read_csv("data/CPIAUCNS.csv", delimiter=',', sep=r', ')
    inflation_df.set_index(["date"], inplace = True)

    for file in os.listdir(json_folder_path):
        json_path = json_folder_path + file
        
        df = json_to_df(json_path)
        df = generate_features(df)
        
        print(df)

        # data = {}
        # data['volume'] = df["volume"]
        # data['20d_vol_avg'] = df["20d_vol_avg"]
        # plot_kp_over_time(data, file)

        create_csv(df, csv_path, file)

if __name__ == "__main__":
    feature_engineering_main()