import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import recall_score

def data_cleaning():
    '''This function cleans the data we will be using
    :return:
    flight_2016: pandas dataframe with the cleaned flight data from 2016
    flight_2017: pandas dataframe with the cleaned fligth data from 2017
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

    flight_2017 = pd.read_csv('fl_samp.csv', delimiter=',')
    #drop useless flight data
    flight_2017.drop(['Year', 'Month', 'Day', 'Flight_Date', 'UniqueCarrier', 'Departure_Time',
                      'Scheduled_Arrival', 'Arrival_Delay', 'Arr_Del_morethan15', 'DistanceGroup',
                      'Carrier_Delay', 'WeatherDelay', 'NAS_Delay', 'Late_Aircraft_Delay',
                      'DEPTIME_GROUP1', 'DEPTIME_GROUP2', 'DEPTIME_GROUP3' ], axis=1, inplace=True)


    return flight_2016, flight_2017

def plot_data():
    '''Creates some plots of the data from 2016 and 2017
    '''
    flight_2016, flight_2017 = data_cleaning()
    #plot 2016 histogram
    fig = plt.figure()
    fig.set_dpi(150)
    plt.hist(flight_2016['Dep_Delay'], color='skyblue', ec='black' )
    plt.title('Departure delay times from 2016')
    plt.show()

    #plot 2016 histogram with log scale
    fig = plt.figure()
    fig.set_dpi(150)
    plt.hist(flight_2016['Dep_Delay'], log=True, bins=10, color='skyblue', ec='black' )
    plt.title('Departure delay times from 2016 log scale')
    plt.show()

    #plot 2017 histogram
    fig = plt.figure()
    fig.set_dpi(150)
    plt.hist(flight_2017['Dep_Delay'], color='skyblue', ec='black' )
    plt.title('Departure delay times from 2017')
    plt.show()

    #plot 2017 histogram with log scale
    fig = plt.figure()
    fig.set_dpi(150)
    plt.hist(flight_2017['Dep_Delay'], log=True, bins=10, color='skyblue', ec='black' )
    plt.title('Departure delay times from 2017 log scale')
    plt.show()

    return

def train_test_data(flight_2016, train_size=0.7, smote=False):
    ''' This function takes in the flight data from 2016 and returns a train_test_split of the data
    :param flight_2016: pandas dataframe containing data
    :param test_size: the amount of data to test on defualts to a 70-30 train test split
    :param smote: parameter to include smote data (to augment points with large delay times that
                  may be infrequent).
    :return X_train, X_test, y_train, y_test:
    '''
    if not smote:

        pass

    else:
        pass



def best_kNN(flight_2016):
    '''Calculates the best hyperparameters for the KNeightborsClassifier, then uses those to
    classify the data
        Parameters:
            flight_2016 (Pandas Dataframe): Any data really, but in this case the flight data
        Returns:
            best_score (float) best accuracy from the data
            recall (recall) best recall score from the data 
            hyperparameters (dictionary) best hyperparameters from the data'''
    X_train,X_test,y_train,y_test = train_test_data(flight_2016)
    neighborclassifier = KNeighborsClassifier()
    parameters = {'n_neighbors':[2,10], 'weights': ('uniform','distance'), 'leaf_size':[20,50], "p":(1,2)}
    gridsearch = GridSearchCV(neighborclassifier, parameters)
    gridsearch.fit(X_train, y_train)
    prediction = gridsearch.predict(X_test)
    best_params = gridsearch.best_params_
    best_score = gridsearch.best_estimator_.score(X_test,y_test)
    recall = recall_score(y_test,prediction)
    return best_score, recall, best_params



#kNN
#NaiveBayes
#RandomForrest
#LogisticRegression
#OLS
#

