import os 
import numpy as np 
import pandas as pd
import numpy_financial as npf

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from arch import arch_model
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from utils import *



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
    
    del df['currency']
    # create_csv(df_types, '', 'dtypes.csv')

    johansen_cointegration_test = coint_johansen(df,-1,5)
    c = {'0.90':0, '0.95':1, '0.99':2}
    traces = johansen_cointegration_test.lr1
    cvts = johansen_cointegration_test.cvt[:, c[str(1-0.05)]]
    def adjust(val, length= 6): 
        return str(val).ljust(length)
    print('Column_Name  >  Test_Stat  >  C(95%)  =>  Signif  \n', '--'*25)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), '  >  ', adjust(round(trace,2), 9), "  >  ", adjust(cvt, 8), '  => ' , trace > cvt)


def calculate_p_value(series):
    ''' Assess col P values to see if theyre suitable for ARIMA
        Link: https://www.pluralsight.com/guides/advanced-time-series-modeling-(arima)-models-in-python 
    '''
    p_value = adfuller(series.dropna())[1]
    print("p-value:", p_value)
    print('Less than 0.05:', 0.05 > p_value)
    


def generate_armina_model(csv_path):

    df = pd.read_csv(csv_path)
    df = df.dropna()
    df.set_index('date', inplace=True)

    cointegration_test_func(df)
    a-b

    # print(df)

    # vol = df['daily_diff']
    # print(vol)
    # calculate_p_value(vol)


    # model = ARIMA(vol, order=(5,1,0))
    # model_fit = model.fit()
    # # summary of fit model
    # print(model_fit.summary())
    # # line plot of residuals
    # residuals = DataFrame(model_fit.resid)
    # residuals.plot()
    # pyplot.show()
    # # density plot of residuals
    # residuals.plot(kind='kde')
    # pyplot.show()
    # # summary stats of residuals
    # print(residuals.describe())


def feature_engineering_main():

    csv_folder_path = "data/csv/" 

    for csv in os.listdir(csv_folder_path):
        csv_path = csv_folder_path + csv
        generate_armina_model(csv_path)

if __name__ == "__main__":
    feature_engineering_main()