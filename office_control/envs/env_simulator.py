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
from scipy.stats import pearsonr
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
from sklearn.model_selection import train_test_split
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from psychrochart.chart import PsychroChart
from matplotlib.lines import Line2D
import statistics
import scipy as sp
import scipy.stats

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


skin_low_limit = 25
skin_high_limit = 37
air_low_limit = 16.5
air_high_limit = 32
humidity_low_limit = 12
humidity_high_limit = 44
clo = {"user1": "0.89", "user2":"1.01", "user3":"1.01", 
"user4":"0.89", "user5":"1.01", "user6": "0.72"}

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
    plt.tight_layout()
    plt.savefig(pngName + "_scatter_size")


def plot_observation(data, pngName, param1, param2, scatter, tag, 
    limit_low, limit_high, limit_low2, limit_high2):
    # plot obervation
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.set_ylim(limit_low, limit_high)
    ax1.plot(range(len(data.index)), data[param1], 'g')
    ax1.set_xlabel('Time (m)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel(param1, color='g')
    ax1.tick_params('y', colors='g')
    if scatter:
        ax2 = ax1.twinx()
        ax2.set_ylim(limit_low2, limit_high2)
        ax2.plot(range(len(data.index)), data[param2], 'r.')
        ax2.set_ylabel(param2, color='r')
        ax2.tick_params('y', colors='r')
    else:
        ax2 = ax1.twinx()
        ax2.set_ylim(limit_low2, limit_high2)
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.plot(range(len(data.index)), data[param2], 'r')
        ax2.set_ylabel(param2, color='r')
        ax2.tick_params('y', colors='r')
    plt.title(tag)
    fig.tight_layout()
    plt.savefig(pngName)
    plt.close()
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_xlim(limit_low, limit_high)
    ax.set_xlabel(param1)
    ax.set_ylim(limit_low2, limit_high2)
    ax.set_ylabel(param2)
    ax.plot(data[param1], data[param2], "r.") 
    plt.title(tag)
    plt.tight_layout()
    plt.savefig(pngName + "_relation")
    plt.close()
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
    def __init__(self, location, env_File, user, input_list, output, filename): 
        self.location = location
        self.user = user
        input_list[0] = user + input_list[0]
        output = user + output
        data_list = input_list + [output]
        self.filename = filename + "_Action4"
        self.data = self.process_data_Action_Air(env_File)[data_list].dropna()
        #self.analyze_data(self.data)
        self.X = self.data[input_list[0:3]]
        print(self.X)
        scalerX = MinMaxScaler()
        scalerX.fit(self.X)
        self.X = scalerX.transform(self.X)
        self.Y = self.data[output].as_matrix().reshape(-1, 1)
        scalerY = MinMaxScaler()
        scalerY.fit(self.Y)
        self.Y= scalerY.transform(self.Y)
        self.X_max = scalerX.data_max_
        self.X_min = scalerX.data_min_
        self.Y_max = scalerY.data_max_
        self.Y_min = scalerY.data_min_
        print(self.X_max, self.X_min)
        print(self.Y_max, self.Y_min)
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(
            self.X, self.Y, test_size=0.4, random_state=42)
        self.output = output
        self.input_list = input_list


    def linearRegression(self):
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
        lr = linear_model.LinearRegression()
        model = lr.fit(self.train_X, self.train_Y)
        #predicted = cross_val_predict(lr, self.X, self.Y, cv=10)

        pred_train_Y = model.predict(self.train_X)
        pred_test_Y = model.predict(self.test_X)
        pred_Y = model.predict(self.X)
        return(pred_train_Y, pred_test_Y, pred_Y)


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
        clf = KernelRidge(alpha=1.0, kernel='rbf', gamma=1.0)
        model = clf.fit(self.train_X, self.train_Y)
        # save the model to disk
        modelname = self.filename + "_" + self.user + "_" + self.output + '_kernel.sav'
        pickle.dump(model, open(modelname, 'wb'))

        pred_train_Y = model.predict(self.train_X)
        pred_test_Y = model.predict(self.test_X)
        pred_Y = model.predict(self.X)
        return(pred_train_Y, pred_test_Y, pred_Y)


    def kNNRegression(self, method):
        """
        method: str, uniform,distance
        """
        # #############################################################################
        # Fit regression model
        n_neighbors = 1

        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=method)
        model = knn.fit(self.train_X, self.train_Y)
        pred_train_Y = model.predict(self.train_X)
        pred_test_Y = model.predict(self.test_X)
        pred_Y = model.predict(self.X)
        return(pred_train_Y, pred_test_Y, pred_Y)


    def SVR(self):
        # Fit regression model
        clf = SVR(kernel='rbf', gamma=1.0)
        model = clf.fit(self.train_X, self.train_Y)
        modelname = self.filename + "_" + self.user + "_" + self.output + '_SVR.sav'
        pickle.dump(model, open(modelname, 'wb'))
        pred_train_Y = model.predict(self.train_X)
        pred_test_Y = model.predict(self.test_X)
        pred_Y = model.predict(self.X)
        return(pred_train_Y, pred_test_Y, pred_Y)


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
        plt.tight_layout()
        plt.savefig(self.filename + "_" + self.output + "_" + "SARIMAXRegression")
        return


    def evaluation(self):
        #Graph
        fig, ax = plt.subplots(figsize=(12,8))

        # Plot data points
        # pd.DataFrame({"Observation": np.concatenate((self.train_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min,
        #  self.test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min), axis=0)}, 
        #     index = range(0, len(self.data.index)*5, 5)).plot(ax=ax, style='o', label='Observed')
        pd.DataFrame({"Observation":self.data[self.output].tolist()}, 
            index = range(0, len(self.data.index)*5, 5)).plot(ax=ax, 
            markersize=10, style='k.', label='Observed')

        
        (kNN_train_test_Y, kNN_pred_test_Y, kNN_pred_Y) = self.kNNRegression("distance")
        MSE_kNN = mean_squared_error(self.test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min,
         kNN_pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min)
        # Plot predictions
        # pd.DataFrame({"kNN": np.concatenate((self.pred_train_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min,
        #  self.pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min), axis=0)}, 
        #     index = range(0, len(self.data.index)*5, 5)).plot(ax=ax, style='g') 
        pd.DataFrame({"kNN": kNN_pred_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min}, 
            index = range(0, len(self.data.index)*5, 5)).plot(ax=ax, style='g')  

        (SVR_train_test_Y, SVR_pred_test_Y, SVR_pred_Y) = self.SVR()
        MSE_SVR = mean_squared_error(self.test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min,
         SVR_pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min)
        # Plot predictions
        # pd.DataFrame({"SVR": np.concatenate((self.pred_train_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min,
        #  self.pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min), axis=0)}, 
        #     index = range(0, len(self.data.index)*5, 5)).plot(ax=ax, style='b') 
        pd.DataFrame({"SVR": SVR_pred_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min}, 
            index = range(0, len(self.data.index)*5, 5)).plot(ax=ax, style='b')  


        (kernel_train_test_Y, kernel_pred_test_Y, kernel_pred_Y) = self.KernelRidgeRegression()
        # calculat MSE only based on test data
        MSE_ker = mean_squared_error(self.test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min, 
            kernel_pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min)
        # Plot predictions
        # pd.DataFrame({"Kernel": np.concatenate((self.pred_train_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min,
        #  self.pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min), axis=0)}, 
        #     index = range(0, len(self.data.index)*5, 5)).plot(ax=ax, style='r')

        pd.DataFrame({"kernel": kernel_pred_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min},
            index = range(0, len(self.data.index)*5, 5)).plot(ax=ax, style='r') 
        
        # for i in range(len(kernel_pred_Y)):
        #     print(self.data[self.input_list[0]][i], self.data[self.output].tolist()[i])
        
        #ax.axvline(len(self.train_Y.flatten())*5, color='k', linestyle='--')

        ax.set(title="Testing Error: (Kernel:%f); (kNN:%f); (SVR:%f)" % (MSE_ker, MSE_kNN, MSE_SVR), xlabel='Time', ylabel=self.output)
        

        # ax2 = ax.twinx()
        # data_m['Action'].plot(ax=ax2, style='g', label='Action')
        legend = ax.legend(loc='upper right')                                                         
        plt.savefig(self.filename + "_" + self.user + "_" + self.output + "_ALL")
        plt.close()

        #plot regression line
        fig, ax = plt.subplots(figsize=(12,6))
        #ax.set_xlim(18, 25)
        ax.set_xlabel('Previous Air Temperature($^\circ$C)')
        #ax.set_ylim(18, 25)
        ax.set_ylabel('Current Air Temperature($^\circ$C)')
        ax.plot(self.data[self.input_list[0]],self.data[self.output],
         "k.", markersize="10", label='Observed')
        ax.plot(self.data[self.input_list[0]] ,
            kernel_pred_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min, 
            "r.", label='Kernel') 
        ax.plot(self.data[self.input_list[0]] ,
            SVR_pred_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min, 
            "b.", label='SVR') 
        ax.plot(self.data[self.input_list[0]] ,
            kNN_pred_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min, 
            "g.", label='kNN') 
        plt.title(self.output)
        legend = ax.legend(loc='lower right') 
        plt.tight_layout()    
        plt.savefig(self.filename + "_" + self.user + "_" + self.output + "_relation")
        plt.close()
        return    


    def process_data_Action_Air(self, datafiles):
        fig, ax1 = plt.subplots(figsize=(12,6))
        ax2 = ax1.twinx()
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

        for i in range(len(datafiles)):
            if i == 0:
                ## action file
                data_action = pd.read_csv(datafiles[i], names = ["Time", "Previous Action"])
                data_action.index = pd.to_datetime(data_action.Time) - pd.Timedelta(hours=5)
                # the action recordeds 5 minutes later 
                data_action.index = data_action.index - pd.Timedelta(minutes = 5)
                # round to closed minutes
                data_action.index = pd.DatetimeIndex(((data_action.index.asi8/(1e9*60)).round()*1e9*60).astype(np.int64))
                ax2.plot(range(0, len(data_action.index)*5, 5), data_action["Previous Action"], 'g.', label = "Action", )
                data = data_action
            else:
                label = datafiles[i].split("-")[1]
                data_env = pd.read_csv(datafiles[i], 
                    names = ["Time", label + " Air Temperature", label + " Relative Humidity"])
                data_env.index = pd.to_datetime(data_env.Time)
                data_env = data_env.resample('30s').mean() # highlight: lable = right, previous 30 second is not good, has too much differences
                data_env_pre = pd.DataFrame(np.array([data_env[(data_env.index == t)].mean()
                for t in data_action.index]), columns = [label + " Previous Air Temperature", 
                label + " Previous Relative Humidity"], index = data_action.index)
                data = pd.concat([data, data_env_pre], axis=1)
                data_env_cur = pd.DataFrame(np.array([data_env[(data_env.index == t + dt.timedelta(minutes = 5))].mean()
                for t in data_action.index]), columns = [label + " Air Temperature", 
                label + " Relative Humidity"], index = data_action.index)
                data = pd.concat([data, data_env_cur], axis=1)
                if(i > 1):
                    ax1.plot(range(0, len(data.index)*5, 5), data[label + " Previous Air Temperature"], label = label)
        legend = ax1.legend(loc='lower right')
        ax1.set_xlabel('Time (m) with 5 minutes interval')
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel("Air Temperature ($^\circ$C)")
        ax2.set_ylabel("Action")
        legend = ax2.legend(loc='upper right')
        plt.title("")
        plt.tight_layout()
        plt.savefig(self.filename + "Air_Temperature_Action_30s")
        plt.close()
        return data


    def analyze_data(self, data):
        data["Delta"] = data[self.user + "Air Temperature"] - data[self.user + "Previous Air Temperature"]
        fig, ax = plt.subplots(figsize=(12,6))
        ax.set_ylabel('Air Temperature 5-minutes Increment ($^\circ$C)')
        ax.set_ylim(-1.0, 1.6)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Action')
        ax.plot(data["Previous Action"], data["Delta"], 
         "r.", label='Observed')
        ax.axhline(0.0, color='k', linestyle='--')
        plt.title("4 Actions for " + self.user)
        plt.savefig(self.filename + self.user + "Actions_relation")
        plt.close()
        color = ["b.", "g.", "m.", "r.", "C3.", "C4.", "C5.", "C1.", "m."]
        fig, ax = plt.subplots(figsize=(12,6))
        ax.xaxis.set_major_locator(MaxNLocator(integer=False))
        ax.set_ylim(-1.0, 1.6)
        #ax.set_xlim(18, 26)
        ax.set_ylabel('Air Temperature 5-minutes Increment ($^\circ$C)')
        ax.set_xlabel('Previous Air Temperature ($^\circ$C)')
        for i in range(4):
            ax.plot(data.loc[data['Previous Action'] == i][self.user + "Previous Air Temperature"],
             data.loc[data['Previous Action'] == i]["Delta"], color[i], markersize="10", label='Action' + str(i))
        legend = ax.legend(loc='lower left')
        ax.axhline(0.0, color='k', linestyle='--')
        plt.savefig(self.filename + self.user + "Previous Air Temperature_Action")
        plt.close()
        fig, ax = plt.subplots(figsize=(12,6))
        ax.xaxis.set_major_locator(MaxNLocator(integer=False))
        ax.set_ylabel('Air Temperature 5-minutes Increment ($^\circ$C)')
        ax.set_xlabel('Outdoor Previous Air Temperature ($^\circ$C)')
        for i in range(4):
            ax.plot(data.loc[data['Previous Action'] == i]["outdoor Previous Air Temperature"],
             data.loc[data['Previous Action'] == i]["Delta"], color[i], markersize="10", label='Action' + str(i))
        legend = ax.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(self.filename + self.user + "Outdoor Air Temperature_Action")
        plt.close()

  



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
    def __init__(self, user, tags, input_list, output): 
        self.user = user
        self.location = "csv/" + user + "/"
        self.tags = tags
        self.filename = self.location + "skin/" + "Skin_" + user
        self.data, self.outdoor = self.combine_data()
        self.analyze_data_Ta_Rh(self.data)
        self.X = self.data[input_list]
        scalerX = MinMaxScaler()
        scalerX.fit(self.X)
        self.X = scalerX.transform(self.X)
        self.Y = self.data['Skin Temperature'].as_matrix().reshape(-1, 1)
        scalerY = MinMaxScaler()
        scalerY.fit(self.Y)
        self.Y= scalerY.transform(self.Y)
        self.Y_max = scalerY.data_max_
        self.Y_min = scalerY.data_min_
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(
            self.X, self.Y, test_size=0.4, random_state=42)
        self.output = output
        self.input_list = input_list

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
        clf = KernelRidge(alpha=1.0, kernel='rbf', gamma=1.0)
        model = clf.fit(self.train_X, self.train_Y)
        # save the model to disk
        modelname = self.filename + "_" + self.output + '_kernel.sav'
        pickle.dump(model, open(modelname, 'wb'))

        pred_train_Y = model.predict(self.train_X)
        pred_test_Y = model.predict(self.test_X)
        pred_Y = model.predict(self.X)
        return(pred_train_Y, pred_test_Y, pred_Y)


    def SVR(self):
        # Fit regression model
        clf = SVR(kernel='rbf', gamma=1.0)
        model = clf.fit(self.train_X, self.train_Y)
        # save the model to disk
        modelname = self.filename + "_" + self.output + '_SVR.sav'
        pickle.dump(model, open(modelname, 'wb'))
        pred_train_Y = model.predict(self.train_X)
        pred_test_Y = model.predict(self.test_X)
        pred_Y = model.predict(self.X)
        return(pred_train_Y, pred_test_Y, pred_Y)


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


    def evaluation(self):
        #Graph
        fig, ax = plt.subplots(figsize=(12,8))

        # Plot observed data points
        pd.DataFrame({"Observation":self.data[self.output].tolist()}, 
            index = range(0, len(self.data.index)*5, 5)).plot(ax=ax, 
           style='k.', label='Observed')

        # SVR 
        (SVR_train_test_Y, SVR_pred_test_Y, SVR_pred_Y) = self.SVR()
        MSE_SVR = mean_squared_error(self.test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min,
         SVR_pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min)

        pd.DataFrame({"SVR": SVR_pred_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min}, 
            index = range(0, len(self.data.index)*5, 5)).plot(ax=ax, style='b')  

        # Kernel
        (kernel_train_test_Y, kernel_pred_test_Y, kernel_pred_Y) = self.KernelRidgeRegression()
        # calculat MSE only based on test data
        MSE_ker = mean_squared_error(self.test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min, 
            kernel_pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min)


        pd.DataFrame({"kernel": kernel_pred_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min},
            index = range(0, len(self.data.index)*5, 5)).plot(ax=ax, style='r') 

        ax.set(title="Testing Error: (Kernel:%f); (SVR:%f)" % (MSE_ker, MSE_SVR), 
            xlabel='Time', ylabel=self.output)
        

        # ax2 = ax.twinx()
        # data_m['Action'].plot(ax=ax2, style='g', label='Action')
        legend = ax.legend(loc='upper right')                                                         
        plt.savefig(self.filename + "_" + self.output + "_ALL")
        plt.close()

        #plot regression line
        fig, ax = plt.subplots(figsize=(12,6))
        ax.set_xlim(air_low_limit, air_high_limit)
        ax.set_xlabel('Air Temperature($^\circ$C)')
        ax.set_ylim(skin_low_limit, skin_high_limit)
        ax.set_ylabel('Skin Temperature($^\circ$C)')
        # ax.tick_params('y', colors='r')
        # ax.tick_params('x', colors='g')
        ax.plot(self.data[self.input_list[0]],self.data[self.output],
         "k.", label='Observed')
        ax.plot(self.data[self.input_list[0]] ,
            kernel_pred_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min, 
            "r.", label='Kernel') 
        ax.plot(self.data[self.input_list[0]] ,
            SVR_pred_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min, 
            "g.", label='SVR') 
        plt.title(self.user)
        legend = ax.legend(loc='lower right') 
        plt.tight_layout()   
        plt.savefig(self.filename + "_" + self.output + "_relation")
        plt.close()
        return    

    def analyze_data_Ta_Rh(self, data):
        #https://matplotlib.org/devdocs/gallery/text_labels_and_annotations/custom_legends.html
        marker_list = ["o", "*", "+", "s", "p", "D"]
        color_list = [  "b", "#02B2FF", "#02FFFF", "#28FF02", "y", "#FF9402", "#FF5602", "r", "#800000"]
        data["Date"] = data.index.date
        output_H = range(28, 37, 1)
        output_L = range(27, 36, 1)
        fig, ax = plt.subplots(figsize=(12,6))
        plot_x = "Air Temperature"
        plot_y = "Relative Humidity"
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(humidity_low_limit, humidity_high_limit)
        ax.set_ylabel("Relative Humidity (%)", fontsize=18)
        ax.set_xlabel('Air Temperature ($^\circ$C)', fontsize=18)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(air_low_limit, air_high_limit + 3)
        plt.xticks(np.arange(17, air_high_limit, 1.0))
        date_list = data["Date"].unique()
        marker_list = ["o", "*", "p", "s", "d", "^"]

        for j in range(len(date_list)):
            for i in range(len(output_H)):
                ax.scatter(data.loc[(data["Skin Temperature"] > output_L[i]) & (data["Skin Temperature"] <= output_H[i])
                 & (data["Date"] == date_list[j])][plot_x], 
                    data.loc[(data["Skin Temperature"] > output_L[i]) & (data["Skin Temperature"] <= output_H[i])
                     & (data["Date"] == date_list[j])][plot_y],
                 color = color_list[i], marker=marker_list[j], label= str(date_list[j]))
      
        plt.title(self.user + ", Clo:" + clo[self.user] + ", Counts:" + str(len(data.index)) ,fontsize=20)
        #legend = ax.legend()
        custom_legend = [Line2D([0], [0], marker= "+", color='#E5E8E8', 
            markerfacecolor='#E5E8E8')] 
        custom_legend = custom_legend + [Line2D([0], [0], marker = m, color='#E5E8E8', 
         markerfacecolor='g', markersize="10") for m in marker_list]
        lengend_list = ["     Outdoor:(Ta,Rh)"]
        lengend_list = lengend_list + [str(date_list[i]) + ":" 
            + str(self.outdoor[i]) for i in range(len(date_list))]
        legend1 = plt.legend(custom_legend, lengend_list, loc='upper right',
         fontsize=12)  

        custom_legend = [Line2D([0], [0], marker= "o",  color='#E5E8E8', 
         markerfacecolor = color, markersize="8") for color in color_list]
        lengend_list = []
        lengend_list = ["27$^\circ$C < Skin <= 28$^\circ$C", "28$^\circ$C < Skin <= 29$^\circ$C", 
        "29$^\circ$C < Skin <= 30$^\circ$C", "30$^\circ$C < Skin <= 31$^\circ$C", 
        "31$^\circ$C < Skin <= 32$^\circ$C", "32$^\circ$C < Skin <= 33$^\circ$C",
        "33$^\circ$C < Skin <= 34$^\circ$C", "34$^\circ$C < Skin <= 35$^\circ$C",
        "35$^\circ$C < Skin <= 36$^\circ$C"]
        plt.legend(custom_legend, lengend_list , loc='lower right',
         fontsize=12) 
        plt.gca().add_artist(legend1)
        plt.tight_layout()
        plt.savefig(self.filename + "_" + plot_x + "_" + plot_y)
        plt.close()
  

    def combine_data(self):
        i = 0
        outdoor_list = []
        for tag in self.tags:
            date = tag[6:]
            skin_File = self.location + "temp-" + tag + ".csv"
            air_File = self.location + "environment-" + tag + ".csv"
            outdoor_File = self.location + "environment-outdoor-" + date + ".csv"
            one_data, output = self.process_data_Skin_Air(skin_File, air_File, outdoor_File)
            outdoor_list.append(output)
            if i == 0:
                data = one_data
            else: 
                data = data.append(one_data)
            plot_observation(one_data, self.location + "skin/" + "Temperature_Humidity_" + tag, 
                 "Air Temperature", "Relative Humidity", False, tag, air_low_limit, air_high_limit, 
                 humidity_low_limit, humidity_high_limit)
            plot_observation(one_data, self.location + "skin/" + "Temperature_Skin_" + tag, 
                "Air Temperature", "Skin Temperature", False, tag, air_low_limit, air_high_limit, 
                 skin_low_limit, skin_high_limit)
            i += 1
        plot_observation(data, self.location + "skin/" + "Temperature_Humidity_" + self.user, 
                 "Air Temperature", "Relative Humidity", False, self.user, air_low_limit, air_high_limit, 
                 humidity_low_limit, humidity_high_limit)
        plot_observation(data, self.location + "skin/" + "Temperature_Skin_" + self.user, 
                "Air Temperature", "Skin Temperature", False, self.user, air_low_limit, air_high_limit, 
                 skin_low_limit, skin_high_limit)
        plot_observation(data, self.location + "skin/" + "Humidity_Skin_" + self.user, 
                "Relative Humidity", "Skin Temperature", False, self.user,  humidity_low_limit, 
                humidity_high_limit, skin_low_limit, skin_high_limit)
        self.plot_3D_observation(data)
        return data, outdoor_list


    def process_data_Skin_Air(self, skin_File, air_File, outdoor_File):
        """"
        Prcoess Air temperature and Relative Humidity 
        Temperature and Humidity is 10s interval, 
        Make every Temperature and Humidity within every 30s as 3 different features 

        Merge Skin and Environment Data
        """
        data_outdoor = pd.read_csv(outdoor_File, names = ["Time", "Outdoor_Temperature", "Outdoor_Humidity"])
        outdoor =  (data_outdoor.mean().round()["Outdoor_Temperature"], data_outdoor.mean().round()["Outdoor_Humidity"])

        ####### process skin temperature
        data_skin = pd.read_csv(skin_File, names = ["Time", "Skin Temperature"])
        data_skin.index = pd.to_datetime(data_skin.Time)
        data_skin = data_skin.resample('30s', closed="right").mean()

        # ####### process air temperature and relative humidity
        data_air = pd.read_csv(air_File, names = ["Time", "Air Temperature", "Relative Humidity"])
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
        return training_set.dropna(), outdoor#.ix[:"2018-02-25 16:25:00"].ix["2018-02-19 21:00:00":]
    


    def plot_3D_observation(self, data):
        ax = plt.figure(figsize=(12,10)).gca(projection='3d')
        ax.scatter(data["Air Temperature"], data["Relative Humidity"], 
            data["Skin Temperature"])
        ax.set_xlabel('Air Temperature')
        ax.set_ylabel('Relative Humidity')
        ax.set_zlabel('Skin Temperature')
        plt.tight_layout()
        plt.savefig(self.filename + "_3D_relation")   
        plt.close()

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
        self.user  = user
        self.filename = self.location + "subjective/" + output + "_" + user
        self.output = output
        data_with_nan, self.outdoor = self.combine_data()
        self.analyze_data_Ta_Rh(data_with_nan)
        if self.output == "Sensation":
            self.analyze_data_scatter(data_with_nan,"Temperature_", air_low_limit, air_high_limit)
            self.analyze_data_scatter(data_with_nan, "Skin_" ,skin_low_limit, skin_high_limit)
            self.analze_data_violin(data_with_nan, "Temperature_0m", air_low_limit, air_high_limit)
            self.analze_data_violin(data_with_nan, "Skin_0m", skin_low_limit, skin_high_limit)
        else:
            self.analyze_data_scatter_h(data_with_nan,"Temperature_", air_low_limit, air_high_limit)
            self.analyze_data_scatter_h(data_with_nan, "Skin_" , skin_low_limit, skin_high_limit)

        ## to get last 5-minutes data will get more nan
        self.data = data_with_nan.dropna()
        self.analyze_data_delta(self.data, 3, "Temperature_")
        self.analyze_data_delta(self.data, 3, "Skin_")
       
        self.X = self.data.ix[:, ["Skin_0m", "Temperature_0m", "Humidity_0m",
              "Skin_5m", "Temperature_5m", "Humidity_5m"]].as_matrix()
        self.Y = np.array(self.data[output].as_matrix()).reshape(-1, 1)
        self.pred_Y = None
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
        model.save_weights(self.location + "subjective/" + self.user + "_"  + self.output + "_ANN.h5")
        self.loss_and_metrics = model.evaluate(X_scaled, dummy_y, batch_size=1)
        classes = model.predict(X_scaled, batch_size=1)
        self.pred_Y = [np.argmax(values) + min(self.Y.flatten()) for values in classes]
        print(min(self.Y.flatten()))


    def evaluation(self, method_name):
        fig, ax = plt.subplots(figsize=(12,8))

        ax.set(title=method_name + ":(categorical_crossentropy:%f; accuracy:%f;)" % (self.loss_and_metrics[0],
         self.loss_and_metrics[1]), xlabel='Time', ylabel=self.output)

        # Plot data points
        pd.DataFrame({"obervation": self.Y.flatten()}).plot(ax=ax, style='b.', markersize=10, label='Observed')

        # Plot predictions with time
        pd.DataFrame({"prediction": self.pred_Y}).plot(ax=ax, style='r.') 
        legend = ax.legend(loc='lower right')    
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(-3.2, 3.2)                                                
        plt.savefig(self.filename + "_" + method_name)
        plt.close()
        fig, ax = plt.subplots(figsize=(12,6))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(-3.2, 3.2)
        ax.set_ylabel('Thermal ' + self.output)
        ax.set_xlabel('Skin Temperature past 5 minutes gradiant($^\circ$C)')
        ax.plot(self.data["Skin_0m"] - self.data["Skin_4m"], self.data[self.output],
         "b.", markersize=10, label='Observation')
        ax.plot(self.data["Skin_0m"] - self.data["Skin_4m"], self.pred_Y,
         "r.", label='prediction')
  
        plt.title(method_name + ":(categorical_crossentropy:%f; accuracy:%f;)" % (self.loss_and_metrics[0],
         self.loss_and_metrics[1]))
        legend = ax.legend(loc='lower right')     
        plt.savefig(self.filename  + "_Delta_" + method_name )
        plt.close()

        self.data["Prediction"] = self.pred_Y
        if(self.output == "Satisfaction"):
            color = ["r", "#EB631B", "#1B86EB", "#58FF33", "#186A06"]
        else:
            color = [ "b", "#1B86EB", "#58FF33", "#EB631B", "r"]
        output = [-2,-1,0,1,2]
        fig, ax = plt.subplots(figsize=(12,6))
        plot_x = "Temperature_0m"
        plot_y = "Humidity_0m"
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        #ax.set_ylim(-3.2, 3.2)
        ax.set_ylabel("Relative Humidity (%)")
        ax.set_xlabel('Air Temperature ($^\circ$C)')
        #ax.set_xlim(18, 31.2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xticks(np.arange(16, air_high_limit + 1, 2.0))
        for i in range(len(output)):
            ax.scatter(self.data.loc[self.data["Prediction"] == output[i]][plot_x], 
                self.data.loc[self.data["Prediction"] == output[i]][plot_y],
             color = color[i], label= self.output + ":" + str(output[i]))
        plt.title(self.user + "_prediction")
        legend = ax.legend(loc='upper left')   
        plt.tight_layout()  
        plt.savefig(self.filename + "_" + plot_x + "_" + plot_y + "_" + method_name)
        plt.close()   


    def mean_confidence_interval(self, data, confidence=0.95):
        a = 1.0*np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
        return m, h


    def analze_data_violin(self, data, input_, limit_low, limit_high):
        output = [ -3, -2,-1, 0, 1, 2, 3]
        fig, ax = plt.subplots(figsize=(12,6))
        plot_x = input_
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(0, 8)
        ax.set_xlabel('Thermal ' + self.output, fontsize=18)
        if input_ == "Temperature_0m": 
            text = "Air"
        elif input_ == "Skin_0m":
            text = "Skin"
        ax.set_ylabel(text + ' Temperature 1 Minutes Average($^\circ$C)' ,fontsize=18)
        ax.set_ylim(limit_low, limit_high)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        #plt.xticks(np.arange(26, 36.2, 1.0))
        all_data = []
        median = []
        mean = []
        errors = []
        for xlabel in output:
            data_x = data.dropna().loc[data.dropna()["Sensation"] == xlabel][plot_x].tolist()
            m, h =  self.mean_confidence_interval(data_x)
            mean.append(m)
            errors.append(h)
            if(len(data_x)  == 0):
                median.append(None)
                all_data.append([0])
            else:
                median.append(statistics.median(data_x))
                all_data.append(data_x)
        ax.violinplot(all_data,
           showmeans=False,
           showmedians=True)
        ax.plot(range(1, 8), median, color="r")
        #ax.plot(range(1, 8), mean, color="b")
        plt.errorbar(range(1, 8), mean, yerr=errors, fmt = 'o', color = 'g')

        # add x-tick labels
        plt.setp(ax, xticks=[y+1 for y in range(len(output))],
             xticklabels=output)
        ax.tick_params(labelsize = 18)
        plt.title(self.user  + ", Clo:" + clo[self.user],  fontsize=20)
        custom_legend = [Line2D([0], [0], marker= "o",  color='g', 
             markerfacecolor = 'g', markersize="6"),
             Line2D([0], [0],  color='r') ]
        lengend_list = ["Mean and 0.95 Confidence Interval", "Median Line"]
        plt.legend(custom_legend, lengend_list , loc='upper left', fontsize=14)   
        plt.tight_layout()
        #legend = ax.legend(loc='lower right')     
        plt.savefig(self.filename + "_" + plot_x + "_violin")
        plt.close()


    def analyze_data_scatter(self, data, input_, limit_low, limit_high, date=""):
        output = [ -3, -2,-1, 0, 1, 2, 3]
        for i in range(6):
            fig, ax = plt.subplots(figsize=(12,6))
            plot_x = input_ + str(i) +"m"
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlim(-3.2, 3.2)
            ax.set_xlabel('Thermal Sensation', fontsize=18)
            if input_ == "Temperature_": 
                text = "Air"
            elif input_ == "Skin_":
                text = "Skin"
            ax.set_ylabel(text + 'Temperature - ' + str(i) + 'minutes($^\circ$C)', fontsize=18)
            ax.set_ylim(limit_low, limit_high)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            #plt.xticks(np.arange(26, 36.2, 1.0))
            correlation = pearsonr(data.dropna()[plot_x], data.dropna()["Sensation"])

            ax.plot(data["Sensation"], data[plot_x], 
             "b.", markersize="9", label='Observed')
            plt.title(self.user + ", Clo:" + clo[self.user] + ", Correlation: " + str(round(correlation[0], 2)), fontsize=20)   
            plt.tight_layout() 
            plt.savefig(self.filename + "_" + plot_x)
            plt.close()


    def analyze_data_scatter_h(self, data, input_, limit_low, limit_high, date=""):
        output = [ -3, -2,-1, 0, 1, 2, 3]
        color_list = [ "b", "#3399ff", "#33ffff",  "#33ff66", "#ff8000", "#ff4000", "#800000"]
        for i in range(6):
            fig, ax = plt.subplots(figsize=(12,6))
            plot_x = input_ + str(i) +"m"
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(-3.2, 3.2)
            ax.set_ylabel('Thermal Satisfaction', fontsize=18)
            if input_ == "Temperature_": 
                text = "Air"
            elif input_ == "Skin_":
                text = "Skin"
            ax.set_xlabel(text + 'Temperature - ' + str(i) + 'minutes($^\circ$C)', fontsize=18)
            ax.set_xlim(limit_low, limit_high)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            #plt.xticks(np.arange(26, 36.2, 1.0))
            correlation = pearsonr(data.dropna()[plot_x], data.dropna()["Satisfaction"])

            for i in range(len(output)):
                ax.scatter(data.loc[(data["Sensation"] == output[i])][plot_x], 
                    data.loc[(data["Sensation"] == output[i])]["Satisfaction"],
                 color = color_list[i])
            plt.title(self.user + ", Clo:" + clo[self.user] + ", Correlation: " + str(round(correlation[0], 2)) , fontsize=20)
            custom_legend = [Line2D([0], [0], marker= "o",  color='#E5E8E8', 
             markerfacecolor = color, markersize="8") for color in color_list]
            lengend_list = ["Cold", "Cool", 
            "Slightly Cool", "Neutral", "Slightly Warm", 
            "Warm", "Hot"]
            plt.legend(custom_legend, lengend_list , loc='upper right',
             fontsize=12)   
            plt.tight_layout()
            plt.savefig(self.filename + "_" + plot_x)
            plt.close()


    def analyze_data_delta(self, data, interval, input_, date=""):
        data["Delta"] = data[input_+"0m"] - data[input_ + str(interval) + "m"]
        output = [-3, -2,-1, 0, 1, 2, 3]
        fig, ax = plt.subplots(figsize=(12,6))
        plot_x = "Delta"
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(-3.2, 3.2)
        ax.set_ylabel('Thermal ' + self.output,  fontsize=18)
        if input_ == "Temperature_": 
                text = "Air"
        elif input_ == "Skin_":
            text = "Skin"
        ax.set_xlabel(text + ' Temperature past '+ str(interval) + ' minutes gradiant($^\circ$C)', fontsize=18)
        correlation = pearsonr(data[plot_x], data[self.output])
        ax.plot(data[plot_x], data[self.output],
         "b.", label='Observed')
  
        plt.title(self.user + "_" + date + ": " + str(round(correlation[0], 2)), fontsize=20)  
        plt.tight_layout()  
        plt.savefig(self.filename + "_" + plot_x + "_" + input_ + "_" + date)
        plt.close()


    def analyze_data_Ta_Rh(self, data, date="", marker="o"):
        #https://matplotlib.org/devdocs/gallery/text_labels_and_annotations/custom_legends.htmlhttps://matplotlib.org/devdocs/gallery/text_labels_and_annotations/custom_legends.html
        if(self.output == "Satisfaction"):
            color_list = ["#800000", "r", "#ff8000", "#66ccff", "#33ff33", "#00cc00", "#006600"]
            offset = 3
        else:
            color_list = [ "b", "#33ffff", "#3399ff", "#33ff66", "#ff8000", "#ff4000", "r"]
            offset = 2
        data["Date"] = data.index.date
        output = [ -3, -2,-1, 0, 1, 2, 3]
        fig, ax = plt.subplots(figsize=(12,6))
        plot_x = "Temperature_0m"
        plot_y = "Humidity_0m"
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(humidity_low_limit, humidity_high_limit)
        ax.set_ylabel("Relative Humidity (%)", fontsize=18)
        ax.set_xlabel('Air Temperature ($^\circ$C)', fontsize=18)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(air_low_limit, air_high_limit + offset)
        plt.xticks(np.arange(17, air_high_limit, 1.0))
        date_list = data["Date"].unique()
        marker_list = ["o", "*", "p", "s", "d", "^"]

        for j in range(len(date_list)):
            for i in range(len(output)):
                ax.scatter(data.loc[(data[self.output] == output[i]) & (data["Date"] == date_list[j])][plot_x], 
                    data.loc[(data[self.output] == output[i]) & (data["Date"] == date_list[j])][plot_y],
                 color = color_list[i], marker=marker_list[j])
        plt.title(self.user + ", Clo:" + clo[self.user] + date + ", Counts:" + str(len(data.index)), fontsize=20)
        #legend = ax.legend()
        custom_legend = [Line2D([0], [0], marker= "+", color='#E5E8E8', 
            markerfacecolor='#E5E8E8')] 
        custom_legend = custom_legend + [Line2D([0], [0], marker= m, color='#E5E8E8', 
         markerfacecolor='g', markersize="10") for m in marker_list]
        lengend_list = ["     Outdoor:(Ta,Rh)"]
        lengend_list = lengend_list + [str(date_list[i]) + ":" 
            + str(self.outdoor[i]) for i in range(len(date_list))]
        legend1 = plt.legend(custom_legend, lengend_list, loc='upper right',
         fontsize=12)  

        custom_legend = [Line2D([0], [0], marker= "o",  color='#E5E8E8', 
         markerfacecolor = color, markersize="8") for color in color_list]
        lengend_list = []
        if self.output == "Satisfaction": 
            lengend_list = ["Strongly Dissatisfied", "Dissatisfied", 
            "Slightly Dissatisfied", "Neutral", "Slightly Satisfied", 
            "Satisfied", "Strongly Satisfied"]
        else: 
            lengend_list = ["Cold", "Cool", 
            "Slightly Cool", "Neutral", "Slightly Warm", 
            "Warm", "Hot"]
        plt.legend(custom_legend, lengend_list , loc='lower right',
         fontsize=12) 
        plt.gca().add_artist(legend1)
        plt.tight_layout()
        plt.savefig(self.filename + "_" + plot_x + "_" + plot_y + "_" + date)
        plt.close()
  


    def combine_data(self):
        i = 0
        outdoor_list = []
        marker_list = ["o", "*", "p", "s", "d", "^"]
        for tag in self.tags:
            date = tag[6:]
            vote_File = self.location + "voting-" + tag + ".csv"
            skin_File = self.location + "temp-" + tag + ".csv"
            air_File = self.location + "environment-" + tag + ".csv"
            outdoor_File = self.location + "environment-outdoor-" + date + ".csv"
            if i == 0:
                data, outdoor = self.process_data_Air_Skin_Sensation(skin_File, air_File, vote_File, outdoor_File)
                #self.analyze_data_Ta_Rh(data, date, marker_list[i])
            
            else:
                data_i, outdoor = self.process_data_Air_Skin_Sensation(skin_File, air_File, vote_File, outdoor_File)
                #self.analyze_data_Ta_Rh(data_i, date, marker_list[i])
                data = data.append(data_i)
            outdoor_list.append(outdoor)
            i += 1
        return data, outdoor_list



    def process_data_Air_Skin_Sensation(self, skin_File, air_File, vote_File, outdoor_File):
        ####### process outdoor

        data_outdoor = pd.read_csv(outdoor_File, names = ["Time", "Outdoor_Temperature", "Outdoor_Humidity"])
        outdoor =  (data_outdoor.mean().round()["Outdoor_Temperature"], data_outdoor.mean().round()["Outdoor_Humidity"])
        ####### process voting
        ### don't do resmaple, since time interval could be different
        data_vote = pd.read_csv(vote_File, names = ["Time", "Sensation", "Satisfaction"])
        data_vote.index = pd.to_datetime(data_vote.Time)
        data_vote.index = pd.DatetimeIndex(((data_vote.index.asi8/(1e9*60)).round()*1e9*60).astype(np.int64))
        data_vote.Time = data_vote.index
        data_vote = data_vote.drop_duplicates()

        ####### process skin temperature
        data_skin = pd.read_csv(skin_File, names = ["Time", "Skin"])
        data_skin.index = pd.to_datetime(data_skin.Time)
        data_skin = data_skin.resample('60s').mean()
        data_skin_0 = pd.DataFrame(np.array([data_skin[(data_skin.index == t )].mean() 
            for t in data_vote.index]), columns = ["Skin_0m"], index = data_vote.index)
        data_skin_1 = pd.DataFrame(np.array([data_skin[(data_skin.index == t - dt.timedelta(minutes = 1))].mean() 
            for t in data_vote.index]), columns = ["Skin_1m"], index = data_vote.index)
        data_skin_2 = pd.DataFrame(np.array([data_skin[(data_skin.index == t - dt.timedelta(minutes = 2))].mean() 
            for t in data_vote.index]), columns = ["Skin_2m"], index = data_vote.index)
        data_skin_3 = pd.DataFrame(np.array([data_skin[(data_skin.index == t - dt.timedelta(minutes = 3))].mean() 
            for t in data_vote.index]), columns = ["Skin_3m"], index = data_vote.index)
        data_skin_4 = pd.DataFrame(np.array([data_skin[(data_skin.index == t - dt.timedelta(minutes = 4))].mean() 
            for t in data_vote.index]), columns = ["Skin_4m"], index = data_vote.index)
        data_skin_5 = pd.DataFrame(np.array([data_skin[(data_skin.index == t - dt.timedelta(minutes = 5))].mean() 
            for t in data_vote.index]), columns = ["Skin_5m"], index = data_vote.index)
        data_vote_skin = data_vote.join(data_skin_0)
        data_vote_skin = data_vote_skin.join(data_skin_1)
        data_vote_skin = data_vote_skin.join(data_skin_2)
        data_vote_skin = data_vote_skin.join(data_skin_3)
        data_vote_skin = data_vote_skin.join(data_skin_4)
        data_vote_skin = data_vote_skin.join(data_skin_5)

        ####### process air temperature and relative humidity
        data_air = pd.read_csv(air_File, names = ["Time", "Temperature", "Humidity"])
        data_air.index = pd.to_datetime(data_air.Time)
        # empty item filled with the value after it
        data_air= data_air.resample('60s').mean()
        #print(data_vote_skin)
        #print(data_air.head(80).index)
        data_air_0 = pd.DataFrame(np.array([data_air[(data_air.index == t)].mean()
            for t in data_vote.index]), columns = ["Temperature_0m", "Humidity_0m"], index = data_vote.index)
        data_air_1 = pd.DataFrame(np.array([data_air[(data_air.index == t - dt.timedelta(minutes = 1))].mean()  
            for t in data_vote.index]), columns = ["Temperature_1m", "Humidity_1m"], index = data_vote.index)
        data_air_2 = pd.DataFrame(np.array([data_air[(data_air.index == t - dt.timedelta(minutes = 2))].mean()  
            for t in data_vote.index]), columns = ["Temperature_2m", "Humidity_2m"], index = data_vote.index)
        data_air_3 = pd.DataFrame(np.array([data_air[(data_air.index == t - dt.timedelta(minutes = 3))].mean()  
            for t in data_vote.index]), columns = ["Temperature_3m", "Humidity_3m"], index = data_vote.index)
        data_air_4 = pd.DataFrame(np.array([data_air[(data_air.index == t - dt.timedelta(minutes = 4))].mean()  
            for t in data_vote.index]), columns = ["Temperature_4m", "Humidity_4m"], index = data_vote.index)
        data_air_5 = pd.DataFrame(np.array([data_air[(data_air.index == t - dt.timedelta(minutes = 5))].mean()  
            for t in data_vote.index]), columns = ["Temperature_5m", "Humidity_5m"], index = data_vote.index)
        data_vote_skin_air = data_vote_skin.join(data_air_0)
        data_vote_skin_air = data_vote_skin_air.join(data_air_1)
        data_vote_skin_air = data_vote_skin_air.join(data_air_2)
        data_vote_skin_air = data_vote_skin_air.join(data_air_3)
        data_vote_skin_air = data_vote_skin_air.join(data_air_4)
        data_vote_skin_air = data_vote_skin_air.join(data_air_5)
      
        return data_vote_skin_air, outdoor



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






def main():
    inSampleTime = '2018-02-10 18:08:00'
    tags = {"user1" : ["user1-2018-2-20", "user1-2018-2-24", "user1-2018-3-2", "user1-2018-3-19"],
    "user2": ["user2-2018-2-20", "user2-2018-2-25", "user2-2018-2-27", "user2-2018-3-2"], 
    "user3": ["user3-2018-2-19", "user3-2018-2-27", "user3-2018-3-2", "user3-2018-3-14", "user3-2018-3-16", "user3-2018-3-17"],
    "user4": ["user4-2018-2-19", "user4-2018-2-25", "user4-2018-3-8", "user4-2018-3-10", "user4-2018-3-20"], 
    "user5": ["user5-2018-3-4", "user5-2018-3-8", "user5-2018-3-10", "user5-2018-3-19", "user5-2018-3-20", "user5-2018-3-22"], 
    "user6": ["user6-2018-2-22", "user6-2018-2-24", "user6-2018-3-4", "user6-2018-3-20", "user6-2018-3-21"]}

    for i in range(1, 7):
        user = "user" + str(i)
        location = "csv/" + user + "/"
        
        skin_simulator = skinSimulator(user, tags[user], ["Air Temperature", "Relative Humidity"], "Skin Temperature")  
        #skin_simulator.evaluation()

        sub_simulator = subjectiveSimulator(user, "Satisfaction", tags[user])
        #sub_simulator.neural_network("adam")
        #sub_simulator.evaluation("neural_network")

        sub_simulator = subjectiveSimulator(user, "Sensation", tags[user])
        # sub_simulator.neural_network("adam")
        # sub_simulator.evaluation("neural_network")
        
    
    # env_File = [location + "action_16.csv",
    #             location + "environment-outdoor-2018-3-6.csv",
    #             location + "environment-user2-2018-3-6.csv",
    #             location + "environment-user4-2018-3-6.csv",
    #             location + "environment-user6-2018-3-6.csv"]
    env_File = [location + "action.csv",
                location + "environment-outdoor-2018-3-13.csv",
                location + "environment-user1-2018-3-13.csv",
                location + "environment-user2-2018-3-13.csv",
                location + "environment-user3-2018-3-13.csv",
                location + "environment-user4-2018-3-13.csv"]
    #data_m = process_data_Sen_Sat(vote_file)
    #plot_sat_sen(data_m, location + "Voting_" + tag, "Satisfaction", "Sensation", tag)
    #plot_relationship(data_m, location + "Voting_" + tag, "Sensation", "Satisfaction")
    #data_m = process_data_Air_Sensation(envir_File,vote_File)
    #plot_observation(data_m, location + "Temperature_Sensation_" + tag, "Temperature", "Sensation", True, tag, -3.5, 3.5)
    #plot_observation(data_m, location + "Temperature_Satisfaction_" + tag, "Temperature", "Satisfaction", False, tag)


    # env_simulator = envSimulator(location, env_File, "user4 ", ["Previous Air Temperature", 
    #     "Previous Action" ,"outdoor Previous Air Temperature"], "Air Temperature", 
    #     location + " envSimulator")
    # env_simulator = envSimulator(location, env_File, "user4 ", ["Previous Relative Humidity", "Previous Action", 
    #     "outdoor Previous Relative Humidity"],
    #   "Relative Humidity", location + "envSimulator")
    # env_simulator.evaluation()
    
    #env_simulator.SARIMAX_prediction()





if __name__ == '__main__':
    main()

