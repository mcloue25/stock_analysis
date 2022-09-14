import os 
import numpy as np 
import pandas as pd
import numpy_financial as npf

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from arch import arch_model
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from utils import *


def dataframe_preprocessing(df):
    ''' # Remove rows with inf's or NaN's & currency
    '''
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis='columns', inplace=True)
    del df['currency']

    return df


def dickey_fuller_assessment(df):

    data = []
    for col_name in df.columns:
        results = {}
        adf, pval, usedlag, nobs, crit_vals, icbest =  adfuller(df[col_name].values)
        results['column'] = col_name
        results['adf'] = adf
        results['nobs'] = nobs
        results['pval'] = pval
        results['icbest'] = icbest
        results['usedlag'] = usedlag
        for key, value in crit_vals.items():
            results['crit_vals_{}'.format(key)] = value

        data.append(results)

    adfuller_df = pd.DataFrame(data)
    adfuller_df.set_index('column', inplace=True)

    print(adfuller_df)

    stationary_cols = adfuller_df.loc[adfuller_df['pval'] < 0.05]
    nonstationary_cols = adfuller_df.loc[adfuller_df['pval'] > 0.05]

    print('STATIONARY COLUMNS')
    print()
    print(stationary_cols)
    print("------------------------")
    print()
    print('NONSTATIONARY COLUMNS')
    print()
    print()
    print(nonstationary_cols)


def create_garch_model():
    ''' 
    '''
    model = arch_model(train, mean='Zero', vol='GARCH', p=15, q=15)
    model_fit = model.fit()


def cointegration_test_func(df): 
    ''' Test if there is a long-run relationship between features
    Args:
        dataframe (float64): Values of the columns to be checked, numpy array of floats 
    
    Returns:
        True or False whether a variable has a long-run relationship between other features
    ''' 
    df_types = df.dtypes
    # print(df_types)
    
    print(df)
    # create_csv(df_types, '', 'dtypes.csv')

    johansen_cointegration_test = coint_johansen(df,-1,5)
    c = {'0.90':0, '0.95':1, '0.99':2}
    traces = johansen_cointegration_test.lr1
    cvts = johansen_cointegration_test.cvt[:, c[str(1-0.05)]]
    def adjust(val, length= 6): 
        return str(val).ljust(length)
    # print('Column_Name  >  Test_Stat  >  C(95%)  =>  Signif  \n', '--'*25)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), '  >  ', adjust(round(trace,2), 9), "  >  ", adjust(cvt, 8), '  => ' , trace > cvt)


def calculate_p_value(series):
    ''' Assess col P values to see if theyre suitable for ARIMA
        Link: https://www.pluralsight.com/guides/advanced-time-series-modeling-(arima)-models-in-python 
    '''
    p_value = adfuller(series.dropna())[1]
    print("p-value:", p_value)
    print('Less than 0.05:', 0.05 > p_value)
    


# def generate_armina_model(df):

#     print(df)

#     vol = df['daily_diff']
#     print(vol)
#     calculate_p_value(vol)


#     model = ARIMA(vol, order=(5,1,0))
#     model_fit = model.fit()
#     # summary of fit model
#     print(model_fit.summary())
#     # line plot of residuals
#     residuals = DataFrame(model_fit.resid)
#     residuals.plot()
#     pyplot.show()
#     # density plot of residuals
#     residuals.plot(kind='kde')
#     pyplot.show()
#     # summary stats of residuals
#     print(residuals.describe())


def assess_feature_worth(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna()
    df.set_index('date', inplace=True)

    df = dataframe_preprocessing(df)

    df = dickey_fuller_assessment(df)

    # cointegration_test_func(df)
    generate_armina_model(df)
    a-b


def feature_engineering_main():

    csv_folder_path = "data/csv/" 

    for csv in os.listdir(csv_folder_path):
        csv_path = csv_folder_path + csv
        assess_feature_worth(csv_path)

if __name__ == "__main__":
    feature_engineering_main()