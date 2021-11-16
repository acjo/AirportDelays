import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KDTree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
import warnings
warnings.filterwarnings('ignore')
#from sklearn.tree import export_graphviz

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
                      'Arrival_Time', 'Dep_Delay', 'Arr_Delay_Minutes',
                      'Arr_Del_morethan15', 'Cancelled', 'Diverted',
                      'DistanceGroup', 'UniqueCarrier', 'Carrier_Delay', 'WeatherDelay', 'NAS_Delay',
                      'Security_Delay', 'Late_Aircraft_Delay', 'Top_Carriers', 'Top_Origin',
                      'DEPTIME_GROUP1', 'DEPTIME_GROUP2', 'DEPTIME_GROUP3' ], axis=1, inplace=True)

    flight_2017 = pd.read_csv('fl_samp.csv', delimiter=',')
    #drop useless flight data
    flight_2017.drop(['Year', 'Month', 'Day', 'Flight_Date', 'UniqueCarrier', 'Departure_Time',
                      'Scheduled_Arrival', 'Dep_Delay', 'Arr_Del_morethan15', 'DistanceGroup',
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
    plt.hist(flight_2016['Arrival_Delay'], color='skyblue', ec='black' )
    plt.title('Arrival delay times from 2016')
    plt.show()

    #plot 2016 histogram with log scale
    fig = plt.figure()
    fig.set_dpi(150)
    plt.hist(flight_2016['Arrival_Delay'], log=True, bins=10, color='skyblue', ec='black' )
    plt.title('Arrival delay times from 2016 log scale')
    plt.show()

    #plot 2017 histogram
    fig = plt.figure()
    fig.set_dpi(150)
    plt.hist(flight_2017['Arrival_Delay'], color='skyblue', ec='black' )
    plt.title('Arrival delay times from 2017')
    plt.show()

    #plot 2017 histogram with log scale
    fig = plt.figure()
    fig.set_dpi(150)
    plt.hist(flight_2017['Arrival_Delay'], log=True, bins=10, color='skyblue', ec='black' )
    plt.title('Arrival delay times from 2017 log scale')
    plt.show()

    return

def smote(X,N,k):
    """ Generate synthetic points using the SMOTE method.
    Parameters:
        X (n,m): minority class samples
        N (int): number of samples to generate from each point
        k (int): number of nearest neighbors
    Returns:
        synthetic ndarray(N*n,m): synthetic minority class samples
    """
    # the number of columns in the number features and
    # the number of rows is the number of observations (points)
    n, m = X.shape
    synthetic_samples = np.zeros((N*n, m))
    #create tree
    tree = KDTree(X)
    for i in range(m):
        #get k nearest neighbors for the current row
        _, indices = tree.query(X[i:i+1], k=k)
        #now we have to create our new samples
        for j in range(N):
            #random choice of the nearest neighbors
            neighbor = X[indices[0][np.random.randint(len(indices[0]))]]
            #now generate a random point that lies between the two original values
            random_point = np.random.uniform(0, 1, m)
            #set the row of the new array
            synthetic_samples[i*N+j] = X[i] + (neighbor - X[i])*random_point

    return synthetic_samples

def train_test_data(train_size=0.7, binary=True, smote_data=True):
    ''' This function takes in the flight data from 2016 and returns a train_test_split of the data
    :param flight_2016: pandas dataframe containing data
    :param train_size: the amount of data to test on defualts to a 70-30 train test split
    :param smote: parameter to include smote data (to augment points with large delay times that
                  may be infrequent).
    :return X_train, X_test, y_train, y_test:
    '''
    flight_2016, _  = data_cleaning()
    #one hot encode
    flight_2016 = pd.get_dummies(flight_2016, columns=['Tai_lNum',
                                                       'Origin_Airport',
                                                       'Origin_City_Name',
                                                       'Origin_State' ], drop_first=True)
    if binary:
        #create the binary labels for if you were late or not

        mask_on_time = flight_2016['Arrival_Delay'] <= 0
        flight_2016 = flight_2016.assign(Delay=lambda x: flight_2016.Arrival_Delay *0)
        flight_2016['Delay'][mask_on_time] = 0
        flight_2016['Delay'][~mask_on_time] = 1
        y = flight_2016['Delay']
        X = flight_2016.drop(['Dep_Delay', 'Delay'], axis=1)
        #traint test split on the binary data
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            train_size=train_size,
                                                            random_state=42)
    else:
        #create appropriate masks
        mask_on_time = flight_2016['Arrival_Delay'] <=0
        mask_15_late = (flight_2016['Arrival_Delay'] > 0) & (flight_2016['Arrival_Delay'] <=15)
        mask_30_late = (flight_2016['Arrival_Delay'] > 15) & (flight_2016['Arrival_Delay'] <=30)
        mask_45_late = (flight_2016['Arrival_Delay'] > 30) & (flight_2016['Arrival_Delay'] <=45)
        mask_60_late = (flight_2016['Arrival_Delay'] > 45) & (flight_2016['Arrival_Delay'] <=60)
        mask_120_late = (flight_2016['Arrival_Delay'] > 60) & (flight_2016['Arrival_Delay'] <=120)
        mask_180_late = (flight_2016['Arrival_Delay'] > 120) & (flight_2016['Arrival_Delay'] <=180)
        mask_240_late = (flight_2016['Arrival_Delay'] > 180) & (flight_2016['Arrival_Delay'] <=240)
        mask_300_late = (flight_2016['Arrival_Delay'] > 240) & (flight_2016['Arrival_Delay'] <=300)
        mask_400_late = (flight_2016['Arrival_Delay'] > 300) & (flight_2016['Arrival_Delay'] <=400)
        mask_500_late = (flight_2016['Arrival_Delay'] > 400) & (flight_2016['Arrival_Delay'] <=500)
        mask_600_late = (flight_2016['Arrival_Delay'] > 500) & (flight_2016['Arrival_Delay'] <=600)
        mask_700_late = (flight_2016['Arrival_Delay'] > 600) & (flight_2016['Arrival_Delay'] <=700)
        mask_800_late = (flight_2016['Arrival_Delay'] > 700) & (flight_2016['Arrival_Delay'] <=800)
        mask_900_late = (flight_2016['Arrival_Delay'] > 800) & (flight_2016['Arrival_Delay'] <=900)
        mask_1000_late = (flight_2016['Arrival_Delay'] > 900) & (flight_2016['Arrival_Delay'] <=1000)
        mask_1000_or_more_late = flight_2016['Arrival_Delay'] > 1000
        flight_2016 = flight_2016.assign(Delay=lambda x: flight_2016.Arrival_Delay *0)

        '''
        masks = [ mask_on_time, mask_15_late, mask_30_late, mask_45_late, mask_60_late, mask_120_late,
                  mask_180_late, mask_240_late, mask_300_late, mask_400_late, mask_500_late, mask_600_late,
                  mask_700_late, mask_800_late, mask_900_late, mask_1000_late, mask_1000_or_more_late]
        times = [0, 15, 30, 40, 60, 120, 180, 240, 300, 400, 500, 600, 700, 800, 900, 1000, 10000]
        for time, mask in zip(times, masks):
            print(time, sum(mask.values))
        '''

        flight_2016['Delay'][mask_on_time] = 0
        flight_2016['Delay'][mask_15_late] = 15
        flight_2016['Delay'][mask_30_late] = 30
        flight_2016['Delay'][mask_45_late] = 40
        flight_2016['Delay'][mask_60_late] = 60
        flight_2016['Delay'][mask_120_late] = 120
        flight_2016['Delay'][mask_180_late] = 180
        flight_2016['Delay'][mask_240_late] = 240
        flight_2016['Delay'][mask_300_late] = 300
        flight_2016['Delay'][mask_400_late] = 400
        flight_2016['Delay'][mask_500_late] = 500
        flight_2016['Delay'][mask_600_late] = 600
        flight_2016['Delay'][mask_700_late] = 700
        flight_2016['Delay'][mask_800_late] = 800
        flight_2016['Delay'][mask_900_late] = 900
        flight_2016['Delay'][mask_1000_late] = 1000
        flight_2016['Delay'][mask_1000_or_more_late] = 10000
        y = flight_2016['Delay']
        X = flight_2016.drop(['Arrival_Delay', 'Delay'], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

        if smote_data:
            #pass
            #add additonal data via SMOTE for everything greater than 240 minutes late
            mask_240 = y_train == 240
            smote_240 = X_train[mask_240]
            additional_smote = smote(smote_240, 26, 2)
            print(additional_smote.size)


    return X_train, X_test, y_train, y_test



def best_kNN(binary):
    '''Calculates the best hyperparameters for the KNeightborsClassifier, then uses those to
    classify the data
        Parameters:
            binary (binary): Use the late binary or not
        Returns:
            best_score (float) best accuracy from the data
            recall (recall) best recall score from the data
            hyperparameters (dictionary) best hyperparameters from the data'''
    X_train,X_test,y_train,y_test = train_test_data(train_size=0.7, binary=binary)
    neighborclassifier = KNeighborsClassifier()
    parameters = {'n_neighbors':[2,10], 'weights': ('uniform','distance'), \
        'leaf_size':[20,50], "p":(1,2), "n_jobs":[-1]}
    gridsearch = GridSearchCV(neighborclassifier, parameters)
    gridsearch.fit(X_train, y_train)
    prediction = gridsearch.predict(X_test)
    best_params = gridsearch.best_params_
    best_score = gridsearch.best_estimator_.score(X_test,y_test)
    recall = recall_score(y_test,prediction)
    return best_score, recall, best_params

def best_logistic(binary):
    '''Calculates the best hyperparameters for the LogisticRegression, then uses those to
    classify the data
        Parameters:
            binary (binary): Use the late binary or not
        Returns:
            best_score (float) best accuracy from the data
            recall (recall) best recall score from the data
            hyperparameters (dictionary) best hyperparameters from the data'''
    X_train,X_test,y_train,y_test = train_test_data(train_size=0.7, binary=binary)
    logisticregression = LogisticRegression()
    parameters = {'penalty':('l1', 'l2', 'elasticnet', 'none'), 'tol':(1e-6,1e-5,1e-4,1e-3,1e-2),\
        'C':(.1,.3,.5,.8,1,1.2,1.5,1.8), "fit_intercept":(False,True), "n_jobs":(-1)}
    gridsearch = GridSearchCV(logisticregression, parameters)
    gridsearch.fit(X_train, y_train)
    prediction = gridsearch.predict(X_test)
    best_params = gridsearch.best_params_
    best_score = gridsearch.best_estimator_.score(X_test,y_test)
    recall = recall_score(y_test,prediction)
    return best_score, recall, best_params

def best_elastic(binary):
    '''Calculates the best hyperparameters for ElasticRegression, then uses those to
    predict the data
        Parameters:
            binary (binary): Use the late binary or not
        Returns:
            best_score (float) best accuracy from the data
            recall (recall) best recall score from the data
            hyperparameters (dictionary) best hyperparameters from the data'''
    X_train,X_test,y_train,y_test = train_test_data(train_size=0.7, binary=binary)
    elastic_regression = ElasticNet()
    parameters = {'alpha':(.5,.8,1,1.2,1.5), 'l1_ratio':(.2,.3,.4,.5,.6,.7,.8),\
        'fit_intercept':(True,False), "normalize":(False,True), "n_jobs":(-1)}
    gridsearch = GridSearchCV(elastic_regression, parameters)
    gridsearch.fit(X_train, y_train)
    prediction = gridsearch.predict(X_test)
    best_params = gridsearch.best_params_
    best_score = gridsearch.best_estimator_.score(X_test,y_test)
    recall = recall_score(y_test,prediction)
    return best_score, recall, best_params

def best_random_forest_reg(binary):
    '''Calculates the best hyperparameters for RandomForestRegression, then uses those to
    predict the data
        Parameters:
            binary (binary): Use the late binary or not
        Returns:
            best_score (float) best accuracy from the data
            recall (recall) best recall score from the data
            hyperparameters (dictionary) best hyperparameters from the data'''
    X_train,X_test,y_train,y_test = train_test_data(train_size=0.7, binary=binary)
    random_forest_regression = RandomForestRegressor()
    parameters = {'n_estimators':(10,50,100,500,1000), 'criterion':("squared_error","absolute_error","poisson"),\
        'max_depth':[5,20], 'bootstrap':(True,False), "n_jobs":(-1)}
    gridsearch = GridSearchCV(random_forest_regression, parameters)
    gridsearch.fit(X_train, y_train)
    prediction = gridsearch.predict(X_test)
    best_params = gridsearch.best_params_
    best_score = gridsearch.best_estimator_.score(X_test,y_test)
    recall = recall_score(y_test,prediction)
    return best_score, recall, best_params

def best_random_forest_class(binary):
    '''Calculates the best hyperparameters for the RandomForestClassifier, then uses those to
    classify the data
        Parameters:
            binary (binary): Use the late binary or not
        Returns:
            best_score (float) best accuracy from the data
            recall (recall) best recall score from the data
            hyperparameters (dictionary) best hyperparameters from the data'''
    X_train,X_test,y_train,y_test = train_test_data(train_size=0.7, binary=binary)
    random_forest_class = RandomForestClassifier()
    parameters = {'n_estimators':(10,50,100,500,1000), 'criterion':("gini", "entropy"),\
        'max_depth':[5,20], 'bootstrap':(True,False), "n_jobs":[-1]}
    gridsearch = GridSearchCV(random_forest_class, parameters)
    gridsearch.fit(X_train, y_train)
    prediction = gridsearch.predict(X_test)
    best_params = gridsearch.best_params_
    best_score = gridsearch.best_estimator_.score(X_test,y_test)
    recall = recall_score(y_test,prediction)
    return best_score, recall, best_params

def best_Gaussian(binary):
    '''Calculates the best hyperparameters for the GaussianNB, then uses those to
    classify the data
        Parameters:
            binary (binary): Use the late binary or not
        Returns:
            best_score (float) best accuracy from the data
            recall (recall) best recall score from the data
            hyperparameters (dictionary) best hyperparameters from the data'''
    X_train,X_test,y_train,y_test = train_test_data(train_size=0.7, binary=binary)
    random_forest_class = GaussianNB()
    parameters = {'var_smoothing': (1e-10,1e-9,1e-8)}
    gridsearch = GridSearchCV(random_forest_class, parameters)
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

if __name__ == "__main__":
    #plot_data()
    #flight_2016, flight_2017 = data_cleaning()
    #print(pd.unique(flight_2016['Dep_Delay']))
    #print(flight_2016['Dep_Delay'].value_counts())
    #X_train, X_test, y_train, y_test = train_test_data(train_size=0.7, binary=False)
    #X_test.iloc[[0, 3]]
    '''
    
    print("Binary")
    print("KNN")
    print(best_kNN(True))
    print("Logistic")
    print(best_logistic(True))
    print("Random Forest Classifer")
    print(best_random_forest_class(True))
    print("Elastic")
    print(best_elastic(True))
    print(best_Gaussian(True))

    print("Not Binary")
    print("KNN")
    print(best_kNN(False))
    print("Random Forest Classifer")
    print(best_random_forest_class(False))
    print("Elastic")
    print(best_elastic(False))
    print("Gaussian")
    print(best_Gaussian(False))
    '''