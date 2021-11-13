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
    #drop useless flight data
    flight_2016.drop(['Month', 'Year', 'Day', 'Flight_Date', 'FlightNum',
                      'Departure_Time', 'DepDel15', 'Dep_Delay_Groups',
                      'Arrival_Time', 'Arrival_Delay', 'Arr_Delay_Minutes',
                      'Arr_Del_morethan15', 'Cancelled', 'Diverted',
                      'DistanceGroup', 'Carrier_Delay', 'WeatherDelay', 'NAS_Delay',
                      'Security_Delay', 'Late_Aircraft_Delay', 'Top_Carriers', 'Top_Origin',
                      'DEPTIME_GROUP1', 'DEPTIME_GROUP2', 'DEPTIME_GROUP3' ], axis=1, inplace=True)

    print(flight_2016.columns)
    #flight_2017 = pd.read_csv('fl_samp.csv', delimiter=',')


    return
data_cleaning()
