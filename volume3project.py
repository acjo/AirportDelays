import numpy as np
import pandas as pd
import sklearn
import scipy

def data_cleaning():
    '''This function cleans the data we will be using
    :return:
    data: pandas dataframe with the cleaned data
    '''
    flight_2016 = pd.read_csv('flight.csv', delimiter=',')
    flight_2017 = pd.read_csv('fl_samp.csv', delimiter=',')


    return