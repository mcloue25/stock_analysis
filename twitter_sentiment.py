import os 
import numpy as np 
import pandas as pd
import numpy_financial as npf

from helpers import *



def create_full_table(company_df, company_tweet_df, tweet_df):

    company_tables_merged = df = pd.merge(company_df, company_tweet_df, how='left', on='ticker_symbol')
    df = pd.merge(company_tables_merged, tweet_df, how='left', on='tweet_id')
    df.set_index(["ticker_symbol"], inplace = True)

    create_csv(df, "data/sentiment_analysis/", "twitter.csv")
    



def sentiment_analysis_main():
    # Only needs to be ran if new Inflation CSV is added 
    # calculate_inflation_rates("data/us_inflation_rates/", "CPIAUCNS.csv")
    company_df = pd.read_csv("data/sentiment_analysis/company.csv", delimiter=',', sep=r', ')
    company_tweet_df = pd.read_csv("data/sentiment_analysis/company_tweet.csv", delimiter=',', sep=r', ')
    tweet_df = pd.read_csv("data/sentiment_analysis/tweet.csv", delimiter=',', sep=r', ')

    # print(tweet_df)

    create_full_table(company_df, company_tweet_df, tweet_df)

if __name__ == "__main__":
    sentiment_analysis_main()