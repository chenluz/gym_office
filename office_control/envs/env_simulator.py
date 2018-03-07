# the actions can be any kinds of control to building
from mpl_toolkits.mplot3d import Axes3D
from pandas import Series
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import neighbors
import statsmodels.api as sm
from datetime import datetime
import requests
from sklearn.metrics import mean_squared_error
import pickle
from matplotlib.ticker import MaxNLocator
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)




# ref: http://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_stata.html
def plot_relationship(data, pngName, param1, param2):
    data.plot(figsize=(12,6))
    plt.savefig(pngName)
    plt.close()
    data.plot.bar(figsize=(12,8))
    plt.savefig(pngName + "_bar")
    plt.close()
    data.plot.scatter(x=param1, y=param2, figsize=(12,8))
    plt.savefig(pngName + "_scatter")
    plt.close()
    return 


def plot_sat_sen(data, pngName, param1, param2, tag):
    fig, ax = plt.subplots(figsize=(12,6))
    two_array  = []
    labels = []
    data_dict = {}
    for item in data[param1].unique():
        labels.append(item)
        two_array.append((data.loc[data['Satisfaction'] == item])['Sensation'].tolist())
    #ax.boxplot(two_array, labels=labels)
    for i in range(-3, 4):
        data_dict[i] = (data.loc[data['Satisfaction'] == i])['Sensation'].tolist()
    size_constant = 20
    for xe, ye in data_dict.items():
        xAxis = [xe] * len(ye)
        #square it to amplify the effect, if you do ye.count(num)*size_constant the effect is barely noticeable
        sizes = [ye.count(num)**2.5 * size_constant for num in ye]
       
        plt.scatter(xAxis, ye, s=sizes)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    #ax.set_xlim(-3, 3)
    #ax.set_ylim(-3, 3)
    ax.set_xlabel('Satisfaction')
    ax.set_ylabel('Sensation')
    plt.title(tag)
    plt.savefig(pngName + "_scatter_size")


def plot_observation(data, pngName, param1, param2, scatter, tag, limit_low, limit_high):
    # plot obervation
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.set_ylim(19, 31)
    ax1.plot(data.index, data[param1], 'g')
    ax1.set_xlabel('Time (m)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel(param1, color='g')
    ax1.tick_params('y', colors='g')
    if scatter:
        ax2 = ax1.twinx()
        ax2.set_ylim(limit_low, limit_high)
        ax2.plot(data.index, data[param2], 'r.')
        ax2.set_ylabel(param2, color='r')
        ax2.tick_params('y', colors='r')
    else:
        ax2 = ax1.twinx()
        ax2.set_ylim(limit_low, limit_high)
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.plot(data.index, data[param2], 'r')
        ax2.set_ylabel(param2, color='r')
        ax2.tick_params('y', colors='r')
    plt.title(tag)
    fig.tight_layout()
    plt.savefig(pngName)
    plt.close()
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_xlim(19, 31)
    ax.set_xlabel('Air Temperature')
    ax.set_ylim(limit_low, limit_high)
    ax.set_ylabel(param2)
    ax.plot(data[param1], data[param2], "r.") 
    plt.title(tag)
    plt.savefig(pngName + "_relation")
    return 

"""
-----------------
Environmental Simulator
-----------------

"""
class envSimulator():
    """
    Simulate air temperature and air humidity according to 
    preivous air temperature and air humidity and actions

    Parameters:
        filename: str, specify the model name and png name that will be saved 

    """
    def __init__(self, location, env_File, input_list, output, filename, inSampleTime): 
        self.location = location
        self.data = process_data_Action_Air(env_File)
        self.train_X = self.data.ix[:inSampleTime, input_list].as_matrix()
        self.train_Y = np.array(self.data.ix[:inSampleTime, output].as_matrix()).reshape(-1, 1) 
        self.test_X = self.data.ix[inSampleTime:, input_list].as_matrix()
        self.test_Y =  np.array(self.data.ix[inSampleTime:, output].as_matrix()).reshape(-1, 1) 
        self.X = self.data[input_list].as_matrix()
        print(self.test_Y)

        # train_X = np.reshape(data_m.ix[:inSampleTime, 'Pre_Temperature'].as_matrix(), (len(train_Y), 1))
        # test_X = np.reshape(data_m['Pre_Temperature'].as_matrix(), 
        #      (len(data_m['Pre_Temperature'].as_matrix()), 1))
        self.output = output
        self.input_list = input_list
        self.filename = filename
        self.inSampleTime = inSampleTime
        self.pred_Y = None
        self.pred_test_Y = None


    def KernelRidgeRegression(self):
        """
        Parameters:
        data_m :dataframe, the dataframe that saved all the all the X and Y 
        input: array, a array of column name in the DataFrame that used as input
        output: str, a column name in the DataFrame that used as output
        filename: str, specify the model name and png name that will be saved 
        inSampleTime: str, the time to split input and output
        """

        # #############################################################################
        # Fit regression model
        clf = KernelRidge(alpha=2.0, kernel='rbf')
        model = clf.fit(self.train_X, self.train_Y)
        # save the model to disk
        modelname = self.filename + "_" + self.output + '_kernel.sav'
        pickle.dump(model, open(modelname, 'wb'))

        self.pred_Y = model.predict(self.X)
        self.pred_test_Y = model.predict(self.test_X)


    def kNNRegression(self, method):
        """
        method: str, uniform,distance
        """
        # #############################################################################
        # Fit regression model
        n_neighbors = 2

        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=method)
        model = knn.fit(self.train_X, self.train_Y)
        self.pred_Y = model.predict(self.X)
        self.pred_test_Y = model.predict(self.test_X)


    def SVR(self):
        # Fit regression model
        clf = SVR(kernel='rbf', gamma=0.1)
        model = clf.fit(self.train_X, self.train_Y)
        self.pred_Y = model.predict(self.X)
        self.pred_test_Y = model.predict(self.test_X)


    def evaluation(self, method_name):
        # calculat MSE only based on test data
        MSE = mean_squared_error(self.test_Y, self.pred_test_Y)

        #Graph
        fig, ax = plt.subplots(figsize=(12,8))

        ax.set(title=method_name + "(MSE:%f)" % MSE, xlabel='Time', ylabel=self.output)

        # Plot data points
        self.data[self.output].plot(ax=ax, style='o', label='Observed')

        # Plot predictions
        pd.DataFrame({method_name: self.pred_Y.flatten()}, index = self.data.index).plot(ax=ax, style='g')  
        ax.axvline(self.inSampleTime, color='k', linestyle='--')

        # ax2 = ax.twinx()
        # data_m['Action'].plot(ax=ax2, style='g', label='Action')

        legend = ax.legend(loc='lower right')      
                                                            
        plt.savefig(self.filename + "_" + self.output + "_" + method_name)


    def SARIMAX_prediction(self):
        # Fit the model
        #The order argument is a tuple of the form (AR specification, Integration order, MA specification)
        # Variables
        initial = self.data.ix[: self.inSampleTime]
        initial_endog = initial[self.input_list[1]]

        initial_exog = sm.add_constant(initial[self.input_list[0]])

        # Fit the model
        mod = sm.tsa.statespace.SARIMAX(initial_endog, exog=initial_exog, order=(1,0,1))
        fit_res = mod.fit(disp=False)
        print(fit_res.summary())

        #http://nbviewer.jupyter.org/gist/ChadFulton/d744368336ef4bd02eadcea8606905b5
        update_endog = self.data[self.output]
        update_exog = sm.add_constant(self.data[self.input_list[0]])
        update_mod = sm.tsa.statespace.SARIMAX(update_endog, exog=update_exog, order=(1,0,1))
        # update_mod.initialize_known(fit_res.predicted_state[:, -length], 
        #     fit_res.predicted_state_cov[:, :, -length])
        update_res = update_mod.filter(fit_res.params)
        print(update_res.summary())
        predict = update_res.get_prediction()
        predict_ci = predict.conf_int()
        startTime = '2018-02-10 14:18:00'

        true_Y =  self.data.ix[self.inSampleTime:, self.output]

        # Dynamic predictions
        predict_dy = update_res.get_prediction(dynamic=self.inSampleTime)
        predict_dy_ci = predict_dy.conf_int()

        MSE = mean_squared_error(true_Y, predict_dy.predicted_mean.ix[self.inSampleTime:])
        #Graph
        fig, ax = plt.subplots(figsize=(12,8))

        ax.set(title='SARIMAX (MES:%f)'% MSE, xlabel='Date', ylabel='Temperature')

        # Plot data points
        self.data.ix[startTime:, self.output].plot(ax=ax, style='o', label='Observed')

        # Plot predictions
        predict_dy.predicted_mean.ix[startTime:].plot(ax=ax, style='g', label='Dynamic forecast ' + self.inSampleTime)
        ci = predict_dy_ci.ix[startTime:]
        ax.fill_between(ci.index, ci.ix[:,0], ci.ix[:,1], color='g', alpha=0.1)

        predict.predicted_mean.ix[startTime:,].plot(ax=ax, style='r--', label='One-step-ahead forecast')
        ci = predict_ci.ix[startTime:,]
        ax.fill_between(ci.index, ci.ix[:,0], ci.ix[:,1], color='r', alpha=0.1)

        legend = ax.legend(loc='lower right')
        plt.savefig(self.filename + "_" + self.output + "_" + "SARIMAXRegression")
        return



"""
-----------------
Occupant Bio Simulator
-----------------

"""
class skinSimulator():
    """
    Simulate skin temperature according to air temperature and air humidity

    Parameters:
        filename: str, specify the model name and png name that will be saved 

    """
    def __init__(self, user, tags): 
        self.user = user
        self.location = "csv/" + user + "/"
        self.tags = tags
        self.filename = self.location + "Skin_" + user
        self.data = self.combine_data()
        self.X = np.array(self.data[["Temperature", "Humidity"]].as_matrix()).reshape(len(self.data), 2)
        self.Y = np.array(self.data["Skin"].as_matrix()).reshape(len(self.data), 1)
        self.pred_Y = None


    def KernelRidgeRegression(self, is_test):
        if is_test == False: 
            # training and saving the modle
            # Fit regression model
            clf = KernelRidge(alpha=2.0, kernel='rbf', gamma=0.1)
            model = clf.fit(self.X, self.Y)
            # save the model to disk
            modlename = self.location + 'Skin_Kernel.sav'
            pickle.dump(model, open(modlename, 'wb'))
        else: 
            # load the trained model 
            model = pickle.load(open(self.location + 'Skin_Kernel.sav', "rb"))
        self.pred_Y = model.predict(self.X)


    def SVR(self, is_test):
        if is_test == False: 
            # training and saving the modle
            # Fit regression model
            clf = SVR(kernel='rbf', gamma=0.1)
            model = clf.fit(self.X, self.Y)
            # save the model to disk
            modlename = self.location + 'Skin_SVR.sav'
            pickle.dump(model, open(modlename, 'wb'))
        else: 
            # load the trained model 
            model = pickle.load(open(self.location + 'Skin_SVR.sav', "rb"))
        self.pred_Y = model.predict(self.X)

        
    def NeuralNetwork(self, is_test):
        # ref:https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/ 
        scalerX = MinMaxScaler()
        scalerX.fit(self.X)
        true_X_scaled = scalerX.transform(self.X)
        
        scalerY = MinMaxScaler()
        scalerY.fit(self.Y)
        true_Y_scaled = scalerY.transform(self.Y)

        model = Sequential()
        model.add(Dense(1, input_dim=1, activation='linear', kernel_initializer='glorot_uniform'))
        model.add(Dense(1, activation='linear'))
        # Compile model
        model.compile(loss='mse',
                  optimizer='sgd',
                  metrics=['accuracy'])
        
        model.fit(true_X_scaled, true_Y_scaled, validation_split=0.2, epochs=200, batch_size=5, verbose=0)
        pred_Y = model.predict(true_X_scaled, batch_size=5)
        self.pred_Y = pred_Y*scalerX.data_max_
    

    def evaluation(self, method_name):
        MSE = mean_squared_error(self.Y, self.pred_Y)

        #Graph
        fig, ax = plt.subplots(figsize=(12,8))

        ax.set(title=method_name + ":(MSE:%f)" % MSE, xlabel='Time', ylabel="Skin Temperature (($^\circ$C))")

        # Plot data points
        pd.DataFrame({"obervation": self.Y.flatten()}).plot(ax=ax, style='o', label='Observed')

        # Plot predictions with time
        pd.DataFrame({"prediction": self.pred_Y.flatten()}).plot(ax=ax, style='g.') 
        legend = ax.legend(loc='lower right')                                                    
        plt.savefig(self.filename + "_" + method_name)
        plt.close()    
        
        # plot regression line
        fig, ax = plt.subplots(figsize=(12,6))
        ax.set_xlim(19, 31)
        ax.set_xlabel('Air Temperature(($^\circ$C))')
        ax.set_ylim(28, 37)
        ax.set_ylabel('Skin Temperature(($^\circ$C))')
        ax.plot(self.data["Temperature"], self.Y, "g.", label='Observed')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(self.data["Temperature"], self.pred_Y.flatten(), "r.", label='Predicted') 
        plt.title(self.user)
        legend = ax.legend(loc='lower right')     
        plt.savefig(self.filename + "_" + method_name + "_relation")      


    def combine_data(self):
        i = 0
        for tag in self.tags:
            skin_File = self.location + "temp-" + tag + ".csv"
            air_File = self.location + "environment-" + tag + ".csv"
            one_data = self.process_data_Skin_Air(skin_File, air_File)
            if i == 0:
                data = one_data
            else: 
                data = data.append(one_data)
            plot_observation(one_data, self.location + "Temperature_Humidity_" + tag, 
                 "Temperature", "Humidity", False, tag, 5, 40)
            plot_observation(one_data, self.location + "Temperature_Skin_" + tag, 
                "Temperature", "Skin", False, tag, 28, 37)
            i += 1
        return data  


    def process_data_Skin_Air(self, skin_File, air_File):
        """"
        Prcoess Air temperature and Relative Humidity 
        Temperature and Humidity is 10s interval, 
        Make every Temperature and Humidity within every 30s as 3 different features 

        Merge Skin and Environment Data
        """

        ####### process skin temperature
        data_skin = pd.read_csv(skin_File, names = ["Time", "Skin"])
        data_skin.index = pd.to_datetime(data_skin.Time)
        data_skin = data_skin.resample('30s', closed="right").mean()

        # ####### process air temperature and relative humidity
        data_air = pd.read_csv(air_File, names = ["Time", "Temperature", "Humidity"])
        data_air.index = pd.to_datetime(data_air.Time)
         # empty item filled with the value after it
        # data_air= data_air.resample('10s').mean().bfill()
        # ## make all the 0/30,10/40 20/50 second as a column,
        # data_air["Second"] = data_air.index.second
        # indices = data_air["Second"] > 29
        # data_air["Second"][indices] = data_air["Second"][indices] - 30  
        # data_air = data_air.pivot(columns='Second')
        # data_air = data_air.resample('30s').mean().dropna()
        data_air = data_air.resample('30s').mean().bfill()
        ###### Merge Skin and Environmental Data and drop wrong data
        training_set = pd.merge(data_skin, data_air, how='inner', left_index=True, right_index=True)
        #print(training_set)
        # index1 = training_set.index.get_loc("2018-02-19 21:00:30" )
        # index2 = training_set.index.get_loc("2018-02-19 21:28:00" )
        # training_set = training_set.drop(training_set.index[range(index1,index2)])
        # with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
        #     print(training_set)
        return training_set.dropna()#.ix[:"2018-02-25 16:25:00"].ix["2018-02-19 21:00:00":]
       

"""
-----------------
Occupant Subjective Simulator
-----------------

"""

class subjectiveSimulator():
    """
    Simulate thermal sensation or thermal satisfaction according to air temperature, air humidity and skin temperature

    Parameters:
        filename: str, specify the model name and png name that will be saved 

    """
    def __init__(self, user, output, tags): 
        self.location = "csv/" + user + "/"
        self.tags = tags
        self.filename = self.location + output + "_" + user
        self.data = self.combine_data()
        self.X = self.data.ix[:, ["Skin_0m", "Temperature_0m", "Humidity_0m",
              "Skin_5m", "Temperature_5m", "Humidity_5m"]].as_matrix()
        self.Y = np.array(self.data[output].as_matrix()).reshape(-1, 1)
        self.pred_Y = None
        self.output = output
        self.loss_and_metrics  = None


    def neural_network(self, optimizer):
        # ref:https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/ 
        ## optimizer options: adam, sgd
        seed(1)
        scalerX = MinMaxScaler()
        scalerX.fit(self.X)
        X_scaled = scalerX.transform(self.X)
        print(scalerX.data_max_)
        print(scalerX.data_min_)
        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(self.Y)
        encoded_Y = encoder.transform(self.Y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_Y)
        lable_num = dummy_y.shape[1]

        model = Sequential()
        model.add(Dense(6, input_dim=6, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(lable_num, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, 
                  metrics=['accuracy'])
        
        model.fit(X_scaled, dummy_y, epochs=200, batch_size=1, verbose=0)
        model.save_weights(self.location + self.output + "_ANN.h5")
        self.loss_and_metrics = model.evaluate(X_scaled, dummy_y, batch_size=1)
        classes = model.predict(X_scaled, batch_size=1)
        self.pred_Y = [np.argmax(values) + min(self.Y.flatten()) for values in classes]
        print(min(self.Y.flatten()))


    def evaluation(self, method_name):
        fig, ax = plt.subplots(figsize=(12,8))

        ax.set(title=method_name + ":(categorical_crossentropy:%f; accuracy:%f;)" % (self.loss_and_metrics[0],
         self.loss_and_metrics[1]), xlabel='Time', ylabel=self.output)

        # Plot data points
        pd.DataFrame({"obervation": self.Y.flatten()}).plot(ax=ax, style='o', label='Observed')

        # Plot predictions with time
        pd.DataFrame({"prediction": self.pred_Y}).plot(ax=ax, style='g.') 
        legend = ax.legend(loc='lower right')                                                    
        plt.savefig(self.filename + "_" + method_name)
        plt.close()     



    def combine_data(self):
        i = 0
        for tag in self.tags:
            vote_File = self.location + "voting-" + tag + ".csv"
            skin_File = self.location + "temp-" + tag + ".csv"
            air_File = self.location + "environment-" + tag + ".csv"
            if i == 0:
                data = self.process_data_Air_Skin_Sensation(skin_File, air_File, vote_File)
            else: 
                data = data.append(self.process_data_Air_Skin_Sensation(skin_File, air_File, vote_File))
            i += 1
        return data



    def process_data_Air_Skin_Sensation(self, skin_File, air_File, vote_File):
        ####### process voting
        ### don't do resmaple, since time interval could be different
        data_vote = pd.read_csv(vote_File, names = ["Time", "Sensation", "Satisfaction"])
        data_vote.index = pd.to_datetime(data_vote.Time)

        ####### process skin temperature
        data_skin = pd.read_csv(skin_File, names = ["Time", "Skin"])
        data_skin.index = pd.to_datetime(data_skin.Time)
        data_skin = data_skin.resample('60s', closed="right").mean()
        data_skin_0 = pd.DataFrame(np.array([data_skin[(data_skin.index == t)].mean() 
            for t in data_vote.index]), columns = ["Skin_0m"], index = data_vote.index)
        data_skin_1 = pd.DataFrame(np.array([data_skin[(data_skin.index == t - dt.timedelta(minutes = 1))].mean() 
            for t in data_vote.index]), columns = ["Skin_1m"], index = data_vote.index)
        data_skin_2 = pd.DataFrame(np.array([data_skin[(data_skin.index == t - dt.timedelta(minutes = 2))].mean() 
            for t in data_vote.index]), columns = ["Skin_2m"], index = data_vote.index)
        data_skin_5 = pd.DataFrame(np.array([data_skin[(data_skin.index == t - dt.timedelta(minutes = 5))].mean() 
            for t in data_vote.index]), columns = ["Skin_5m"], index = data_vote.index)
        data_vote_skin = data_vote.join(data_skin_0)
        data_vote_skin = data_vote_skin.join(data_skin_1)
        data_vote_skin = data_vote_skin.join(data_skin_2)
        data_vote_skin = data_vote_skin.join(data_skin_5)

        ####### process air temperature and relative humidity
        data_air = pd.read_csv(air_File, names = ["Time", "Temperature", "Humidity"])
        data_air.index = pd.to_datetime(data_air.Time)
        # empty item filled with the value after it
        data_air= data_air.resample('60s').mean().bfill()
        data_air_0 = pd.DataFrame(np.array([data_air[(data_air.index == t)].mean()
            for t in data_vote.index]), columns = ["Temperature_0m", "Humidity_0m"], index = data_vote.index)
        data_air_1 = pd.DataFrame(np.array([data_air[(data_air.index == t - dt.timedelta(minutes = 1))].mean()  
            for t in data_vote.index]), columns = ["Temperature_1m", "Humidity_1m"], index = data_vote.index)
        data_air_2 = pd.DataFrame(np.array([data_air[(data_air.index == t - dt.timedelta(minutes = 2))].mean()  
            for t in data_vote.index]), columns = ["Temperature_2m", "Humidity_2m"], index = data_vote.index)
        data_air_5 = pd.DataFrame(np.array([data_air[(data_air.index == t - dt.timedelta(minutes = 5))].mean()  
            for t in data_vote.index]), columns = ["Temperature_5m", "Humidity_5m"], index = data_vote.index)
        data_vote_skin_air = data_vote_skin.join(data_air_0)
        data_vote_skin_air = data_vote_skin_air.join(data_air_1)
        data_vote_skin_air = data_vote_skin_air.join(data_air_2)
        data_vote_skin_air = data_vote_skin_air.join(data_air_5)
        return data_vote_skin_air.dropna()



"""
-----------------
Individual Thermal Model
-----------------

"""


def neural_network_prediction(data_m, filename, is_test):
    # ref:https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/ 
    true_X = data_m.ix[:, 3:12].as_matrix()
    true_Y = np.array(data_m["Sensation"].as_matrix()).reshape(len(data_m), 1)
    scalerX = MinMaxScaler()
    scalerX.fit(true_X)
    true_X_scaled = scalerX.transform(true_X)
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(true_Y)
    encoded_Y = encoder.transform(true_Y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
   # print(dummy_y)

    model = Sequential()
    model.add(Dense(9, input_dim=9, activation='relu'))
    model.add(Dense(14, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    model.fit(true_X_scaled, dummy_y, epochs=200, batch_size=5, verbose=0)
    classes = model.predict(true_X_scaled, batch_size=5)

    print([np.argmax(values)-1 for values in classes])



def process_data_Sen_Sat(vote_file):
    data_vote = pd.read_csv(vote_file)
    data_vote.index = pd.to_datetime(data_vote.Time)
    data_vote = data_vote.resample('300s').mean()
    return data_vote.dropna()



def process_data_Air_Sensation(Envir_File, vote_file):
    """"
    Prcoess Air temperature and Relative Humidity 
    Temperature and Humidity is 10s interval, 

    """

    ####### process voting 
    data_vote = pd.read_csv(vote_file)
    data_vote.index = pd.to_datetime(data_vote.Time)
    data_vote = data_vote.resample('10s').mean()

    ####### process air temperature and relative humidity
    data_air = pd.read_csv(Envir_File)
    data_air.index = pd.to_datetime(data_air.Time)
     # empty item filled with the value after it
    data_air= data_air.resample('10s').mean().bfill()
    ###### Merge Skin and Environmental Data and drop wrong data
    training_set = pd.merge(data_vote, data_air, how='outer', left_index=True, right_index=True)
    return training_set



def process_data_Action_Air(filename):
    data = pd.read_csv(filename)
    # Dataset
    data.index = pd.to_datetime(data.Time)
    return data


def main():
    inSampleTime = '2018-02-10 18:08:00'
    ###### user3 ######
    # combine all the files for user3 
    location = "csv/env/"
    user = "user3"
    tags = ["user3-2018-2-19", "user3-2018-2-27", "user3-2018-3-2"]
    env_File = location + "environment.csv"
    #data_m = process_data_Sen_Sat(vote_file)
    #plot_sat_sen(data_m, location + "Voting_" + tag, "Satisfaction", "Sensation", tag)
    #plot_relationship(data_m, location + "Voting_" + tag, "Sensation", "Satisfaction")
    #data_m = process_data_Air_Sensation(envir_File,vote_File)
    #plot_observation(data_m, location + "Temperature_Sensation_" + tag, "Temperature", "Sensation", True, tag, -3.5, 3.5)
    #plot_observation(data_m, location + "Temperature_Satisfaction_" + tag, "Temperature", "Satisfaction", False, tag)


    # env_simulator = envSimulator(location, env_File, ["Action", "Pre_Temperature"],
    #  "Temperature", location + "envSimulator", inSampleTime)
    # # env_simulator = envSimulator(location, env_File, ["Action", "Pre_Humidity"],
    # #  "Humidity", location + "envSimulator", inSampleTime)
    # env_simulator.KernelRidgeRegression()
    # # env_simulator.kNNRegression("distance")
    # # env_simulator.SVR()
    # env_simulator.evaluation("RBFKernel2")
    # env_simulator.SARIMAX_prediction()

    skin_simulator = skinSimulator(user, tags)
    skin_simulator.KernelRidgeRegression(False)
    skin_simulator.evaluation("Kernel")

    # sub_simulator = subjectiveSimulator(user, "Satisfaction", tags)
    # sub_simulator.neural_network("adam")
    # sub_simulator.evaluation("neural_network")

if __name__ == '__main__':
    main()

