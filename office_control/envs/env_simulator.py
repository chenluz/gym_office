# the actions can be any kinds of control to building
from mpl_toolkits.mplot3d import Axes3D
from pandas import Series
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import numpy as np
import pandas as pd
from scipy.stats import norm, halfnorm, truncnorm, multivariate_normal
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
import math
import humidity 
import seaborn as sns
import plotly.plotly as py
import plotly.figure_factory as ff
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

from numpy.random import seed
import json
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


skin_low_limit = 25
skin_high_limit = 37
air_low_limit = 16.5
air_high_limit = 32
Rh_low_limit = 12
Rh_high_limit = 44
humidity_low_limit = 0.001
humidity_high_limit = 0.008
Rh_out_low_limit = 17
Rh_out_high_limit = 82

humidity_out_low_limit = 0.001
humidity_out_high_limit = 0.015

Ta_out_low_limit = 0
Ta_out_high_limit = 21

hrate_low_limit = 54
hrate_high_limit= 93

rr_low_limit = 0.539
rr_high_limit = 1.04

skin3m_low_limit = 0 
skin3m_high_limit = 0.75

skin5m_low_limit = 0
skin5m_high_limit = 1.3

Air3m_low_limit = 0 
Air3m_high_limit = 1.06

Air5m_low_limit = 0
Air5m_high_limit = 1.7
clo = {"user1": "1.0", "user2":"1.0", "user3":"1.0", 
"user4":"1.0", "user5":"1.0", "user6": "0.72"}
# Rh_in_limit = {"user4":[14.8, 42.5]}  # for indoor humidity prediction
# Rh_out_limit = {"user4":[25.95, 82.3]} # for indoor humidity prediction
# Ta_Rh_limit = {"user4":[]} # 
#  Ta_Skin_limit = {"user4": [1,1]} # for skin temperature prediciton
# Rh_Skin_limit = {"user4": [1,1]} # for skin temperature prediciton
# Ta_Vote_limit =  {"user4": [17.05,30.78]}   # for voting prediciton
# Rh_Vote_limit =  {"user4": [15.1,36.133]}   # for voting prediciton
# Skin_limit = {"user4":[26.106, 35.88]} # for voting prediction

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
class TempSimulator():
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
Relative Humidity Simulator
-----------------

"""
class HumiditySimulator():
    """
    Simulate relative humidity according to indoor air temperature and outdoor air humidity 

    """
    def __init__(self, user, tags, simulation=False): 
        self.location = "csv/" + user + "/"
        self.tags = tags
        self.user  = user
        self.output = "RH"
        self.filename = self.location + "environment/temperature" + "_" + user      
        data_with_nan = self.combine_data()
        self.data = data_with_nan.dropna()
        self.simulation = simulation
        self.X = self.data[["Temperature", "Out_RH"]].as_matrix()
        
        self.Y = self.data[self.output ].as_matrix().reshape(-1, 1)
        self.X_max = np.asarray([air_high_limit, Rh_out_high_limit]) 
        self.X_min = np.asarray([air_low_limit, Rh_out_low_limit]) 
        self.X = (self.X - self.X_min)/(self.X_max - self.X_min)
        self.Y_max = np.asarray([Rh_high_limit])
        self.Y_min = np.asarray([Rh_low_limit])
        self.Y = (self.Y - self.Y_min)/(self.Y_max - self.Y_min)

        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(
            self.X, self.Y, test_size=0.3, random_state=42)
        if self.simulation == True: 
            self.simulatedData = {
            "Ta": (np.random.choice(np.arange(self.X_min[0], self.X_max[0], 0.5), 1000) 
                    - self.X_min[0])/(self.X_max[0] - self.X_min[0]),
            # "Out_Humidity": (np.random.choice(np.arange(self.X_min[1], self.X_max[1], 0.0002), 1000) 
            #     - self.X_min[1])/(self.X_max[1] - self.X_min[1]),
            "Out_Humidity": (np.random.choice(np.arange(self.X_min[1], self.X_max[1], 0.5), 1000) 
                 - self.X_min[1])/(self.X_max[1] - self.X_min[1]),
            }
            self.simulatedDf = pd.DataFrame(self.simulatedData).ix[:,["Ta", "Out_Humidity"]]
            self.test_X = self.simulatedDf.as_matrix()
           
            


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
        if self.simulation == False:
            model = clf.fit(self.train_X, self.train_Y)
            # save the model to disk
            modelname = self.filename + "_" + self.output + '_kernel.sav'
            pickle.dump(model, open(modelname, 'wb'))
        else:
            # load model from disk
            model = pickle.load(open(self.filename + "_RH_kernel.sav", 'rb'))

        pred_train_Y = model.predict(self.train_X)
        pred_test_Y = model.predict(self.test_X)
        pred_Y = model.predict(self.X)
        return(pred_train_Y, pred_test_Y, pred_Y)


    def SVR(self):
        # Fit regression model
        clf = SVR(kernel='rbf', gamma=1.0)
        if self.simulation == False:
            model = clf.fit(self.train_X, self.train_Y)
            # save the model to disk
            modelname = self.filename + "_" + self.output + '_SVR.sav'
            pickle.dump(model, open(modelname, 'wb'))
        else:
            # load model from disk
            model = pickle.load(open(self.filename + "_RH_SVR.sav", 'rb'))

        pred_train_Y = model.predict(self.train_X)
        pred_test_Y = model.predict(self.test_X)
        pred_Y = model.predict(self.X)
        return(pred_train_Y, pred_test_Y, pred_Y)


    def evaluation(self):
        if self.simulation == False:
            fig, ax = plt.subplots(figsize=(12,8))

            # Plot observed data points
            # pd.DataFrame({"Outdoor Humidity":self.data["Out_Humidity"].tolist()}, 
            #     index = range(0, len(self.data.index))).plot(ax=ax, 
            #    style='b', label='Outdoor Humidity')

            # Plot observed data points
            pd.DataFrame({"Indoor Humidity":self.data[self.output].tolist()}, 
                index = range(0, len(self.data.index))).plot(ax=ax, 
               style='k', label='Indoor Humidity')

            # SVR 
            (SVR_train_test_Y, SVR_pred_test_Y, SVR_pred_Y) = self.SVR()
            MSE_SVR = mean_squared_error(self.test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min,
             SVR_pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min)

            pd.DataFrame({"SVR": SVR_pred_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min}, 
                index = range(0, len(SVR_pred_Y.flatten()))).plot(ax=ax, style='g')  

            # Kernel
            (kernel_train_test_Y, kernel_pred_test_Y, kernel_pred_Y) = self.KernelRidgeRegression()
            # calculat MSE only based on test data
            MSE_ker = mean_squared_error(self.test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min, 
                kernel_pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min)


            pd.DataFrame({"kernel": kernel_pred_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min},
                index = range(0, len(kernel_pred_Y.flatten()))).plot(ax=ax, style='r') 

            plt.title("Testing Error: (Kernel:%f); (SVR:%f)" % (MSE_ker, MSE_SVR), fontsize=20)
            ax.set_xlabel('Minutes', fontsize=18)
            #ax.set_ylabel('Humidity Ratio (kg/kg)', fontsize=18)
            ax.set_ylabel('Relative Humidity(%)', fontsize=18)
            ax.tick_params(labelsize = 18)
            

            # ax2 = ax.twinx()
            # data_m['Action'].plot(ax=ax2, style='g', label='Action')
            legend = ax.legend(loc='upper right') 
            plt.tight_layout()                                                        
            plt.savefig(self.filename + "_" + self.output + "_ALL")
            plt.close()

        if self.simulation == True:
            # ref: https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
            (kernel_train_test_Y, kernel_pred_test_Y, kernel_pred_Y) = self.KernelRidgeRegression()
            test_X = self.test_X*(self.X_max - self.X_min) + self.X_min
            test_df = pd.DataFrame(test_X, columns = ["Temperature", "Humidity_Out"])
            test_df["Humidity_In"] = kernel_pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min
            fig, ax = plt.subplots(figsize=(12,8))
            ax = plt.axes(projection='3d')
            ax.scatter3D(test_df["Temperature"], test_df["Humidity_Out"], 
                test_df["Humidity_In"], c=test_df["Humidity_In"], cmap='Blues');
            # ax.set_zlabel("Indoor Humidity Ratio(kg/kg)", fontsize=14)
            # ax.set_zlim(humidity_low_limit, humidity_high_limit)
            # ax.set_ylabel("Outdoor Humidity Ratio(kg/kg)", fontsize=14)
            # ax.set_ylim(humidity_out_low_limit, humidity_out_high_limit)
            ax.set_zlabel("Indoor Relative Humidity (%)", fontsize=14)
            ax.set_zlim(Rh_low_limit, Rh_high_limit)
            ax.set_ylabel("Outdoor Relative Humidity(%)", fontsize=14)
            ax.set_ylim(Rh_out_low_limit, Rh_out_high_limit)
            ax.set_xlabel("Indoor Temperature($^\circ$C)", fontsize=14)
            ax.set_xlim(air_low_limit, air_high_limit)
            ax.set_facecolor('xkcd:salmon')
            ax.tick_params(labelsize = 12)
            plt.title(self.user + " predicted indoor humidity using randomly generated data", fontsize=16)
            plt.tight_layout()
            plt.savefig(self.filename + "_" + self.output + "_3DSimulate")
        else:
            fig, ax = plt.subplots(figsize=(12,8))
            ax = plt.axes(projection='3d')
            # ax.scatter3D(self.data["Temperature"], self.data["Out_Humidity"], 
            #     self.data["Humidity"], c=self.data["Humidity"], cmap='Blues');
            # ax.set_zlabel("Indoor  Humidity Ratio(kg/kg)", fontsize=14)
            # ax.set_zlim(humidity_low_limit, humidity_high_limit)
            # ax.set_ylabel("Outdoor Humidity Ratio(kg/kg)", fontsize=14)
            # ax.set_ylim(humidity_out_low_limit, humidity_out_high_limit)
            ax.scatter3D(self.data["Temperature"], self.data["Out_RH"], 
                self.data["RH"], c=self.data["RH"], cmap='Blues');
            ax.set_zlabel("Indoor Relative Humidity (%)", fontsize=14)
            ax.set_zlim(Rh_low_limit, Rh_high_limit)
            ax.set_ylabel("Outdoor Relative Humidity (%)", fontsize=14)
            ax.set_ylim(Rh_out_low_limit, Rh_out_high_limit)
            ax.set_xlabel("Indoor Temperature($^\circ$C)", fontsize=14)
            ax.set_xlim(air_low_limit, air_high_limit)
            ax.set_facecolor('xkcd:salmon')
            ax.tick_params(labelsize = 12)
            plt.title(self.user + " Real indoor humidity", fontsize=16)
            plt.tight_layout()
            plt.savefig(self.filename + "_" + self.output + "_3D")



    def combine_data(self):
        i = 0
        for tag in self.tags:
            date = tag[6:]
            air_File = self.location + "environment-" + tag + ".csv"
            outdoor_File = self.location + "environment-outdoor-" + date + ".csv"
            if i == 0:
                data = self.process_data_Air_Rh(air_File, outdoor_File)
            
            else:
                data_i = self.process_data_Air_Rh(air_File, outdoor_File)
                data = data.append(data_i)
            i += 1
        return data



    def process_data_Air_Rh(self, air_File, outdoor_File):
        ####### process indoor air temperature and relative humidity
        data_air = pd.read_csv(air_File, names = ["Time", "Temperature", "RH"])
        data_air.index = pd.to_datetime(data_air.Time)
        data_air["Humidity"] = get_humidity_ratio(data_air["Temperature"].tolist(), data_air["RH"].tolist())
        # empty item filled with the value after it
        data_air= data_air.resample('30s').mean()
        ##### process outdoor air temperature and relative humidity 
        data_outdoor = pd.read_csv(outdoor_File, names = ["Time", "Out_Temperature", "Out_RH"])
        data_outdoor.index = pd.to_datetime(data_outdoor.Time)
        data_outdoor = data_outdoor.resample('30s').mean()
        data_outdoor_0 = pd.DataFrame(np.array([data_outdoor[(data_outdoor.index == t)].mean()
            for t in data_air.index]), columns = ["Out_Temperature", "Out_RH"], index = data_air.index)
        data_air["Out_Humidity"] = get_humidity_ratio(data_outdoor_0["Out_Temperature"].tolist(), data_outdoor_0["Out_RH"].tolist())
        data_air_out = data_air.join(data_outdoor_0)
      
        return data_air_out


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
    def __init__(self, user, tags, input_list, output, simulation = False): 
        self.user = user
        self.location = "csv/" + user + "/"
        self.tags = tags
        self.output = output
        self.input_list = input_list
        self.simulation = simulation
        self.filename = self.location + "skin/" + "Skin_" + user
        self.data, self.outdoor = self.combine_data()
        if self.simulation == False:
            self.analyze_data_Ta_Rh(self.data)
        self.X = self.data[input_list].as_matrix()
        self.Y = self.data['Skin Temperature'].as_matrix().reshape(-1, 1)
        self.X_max = np.asarray([air_high_limit, Rh_high_limit])
        self.X_min = np.asarray([air_low_limit, Rh_low_limit])
        self.X = (self.X - self.X_min)/(self.X_max - self.X_min)
        self.Y_max = np.asarray([skin_high_limit])
        self.Y_min = np.asarray([skin_low_limit])
        self.Y = (self.Y - self.Y_min)/(self.Y_max - self.Y_min)
        
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(
            self.X, self.Y, test_size=0.3, random_state=42)
        if self.simulation == True: 
            # predict humidity based on random selected indoor tempeature and outdoor humidity in certrain ranges 
            model = pickle.load(open(self.location + "environment/temperature_" + self.user + "_RH_kernel.sav", 'rb'))
            Ta = (np.random.choice(np.arange(air_low_limit, air_high_limit, 0.5), 1000) 
                - air_low_limit)/(air_high_limit - air_low_limit)
            self.simulatedData1 = {
            "Ta": Ta,
            "Rh_out": (np.random.choice(np.arange(Rh_out_low_limit, Rh_out_high_limit, 0.5),
             1000) - Rh_out_low_limit)/(Rh_out_high_limit - Rh_out_low_limit)
            }
            self.simulatedDf1 = pd.DataFrame(self.simulatedData1).ix[:,["Ta","Rh_out"]]
            self.test_X1= self.simulatedDf1.as_matrix()
            humidity = model.predict(self.test_X1).flatten()
            # combine random generated indoor temperature and predicted humidity for skin temperature prediciton
            self.simulatedData = {
            "Ta": Ta,
            "Rh": humidity
            }
            self.simulatedDf = pd.DataFrame(self.simulatedData).ix[:,["Ta", "Rh"]]
            self.test_X = self.simulatedDf.as_matrix()


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
        if self.simulation == False:
            model = clf.fit(self.train_X, self.train_Y)
            # save the model to disk
            modelname = self.filename + "_" + self.output + '_SVR.sav'
            pickle.dump(model, open(modelname, 'wb'))
        else:
            # load model from disk
            model = pickle.load(open(self.filename + "_Skin Temperature_SVR.sav", 'rb'))

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
        if self.simulation == False:
            fig, ax = plt.subplots(figsize=(12,8))

            # Plot observed data points
            pd.DataFrame({"Observation":self.data[self.output].tolist()}, 
                index = range(0, len(self.data.index))).plot(ax=ax, 
               style='k', label='Observed')

            # SVR 
            (SVR_train_test_Y, SVR_pred_test_Y, SVR_pred_Y) = self.SVR()
            MSE_SVR = mean_squared_error(self.test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min,
             SVR_pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min)

            pd.DataFrame({"SVR": SVR_pred_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min}, 
                index = range(0, len(SVR_pred_Y.flatten()))).plot(ax=ax, style='g')  

            # Kernel
            (kernel_train_test_Y, kernel_pred_test_Y, kernel_pred_Y) = self.KernelRidgeRegression()
            # calculat MSE only based on test data
            MSE_ker = mean_squared_error(self.test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min, 
                kernel_pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min)


            pd.DataFrame({"kernel": kernel_pred_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min},
                index = range(0, len(kernel_pred_Y.flatten()))).plot(ax=ax, style='r') 

            plt.title("Testing Error: (Kernel:%f); (SVR:%f)" % (MSE_ker, MSE_SVR), fontsize=20)
            ax.set_xlabel('Minutes', fontsize=18)
            ax.set_ylabel('Skin Temperature ($^\circ$C)', fontsize=18)
            ax.tick_params(labelsize = 18)

            legend = ax.legend(loc='upper right') 
            plt.tight_layout()                                                        
            plt.savefig(self.filename + "_" + self.output + "_ALL")
            plt.close()
        else:
            (SVR_train_test_Y, SVR_pred_test_Y, SVR_pred_Y) = self.SVR()

        #plot regression line
        fig, ax = plt.subplots(figsize=(12,6))
        ax.set_xlim(air_low_limit, air_high_limit)
        ax.set_xlabel('Air Temperature($^\circ$C)', fontsize=18)
        ax.set_ylim(skin_low_limit, skin_high_limit)
        ax.set_ylabel('Skin Temperature($^\circ$C)', fontsize=18)
        # ax.tick_params('y', colors='r')
        # ax.tick_params('x', colors='g')
        XX = self.test_X[:,0]*(self.X_max[0] - self.X_min[0]) + self.X_min[0]

        if self.simulation == False:
            ax.plot(XX,self.test_Y*(self.Y_max - self.Y_min) + self.Y_min,
                "k.", label='Observed')
            ax.plot(XX,
                kernel_pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min, 
                "r.", label='Kernel') 
        ax.scatter(XX,
            SVR_pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min, 
            c=self.test_X[:,1]*(self.X_max[1] - self.X_min[1]) + self.X_min[1], cmap="Blues", label='SVR') 
        ax.tick_params(labelsize = 18)
        plt.title(self.user, fontsize=20)
        legend = ax.legend(loc='lower right') 
        plt.tight_layout()   
        if self.simulation == False:
            plt.savefig(self.filename + "_" + self.output + "_relation")
        else:
            plt.savefig(self.filename + "_" + self.output + "_relation_simulation") 
        plt.close()

        test_X = self.test_X*(self.X_max - self.X_min) + self.X_min
        test_df = pd.DataFrame(test_X, columns = ["Air Temperature", "Relative Humidity"])
        test_df["Skin Temperature"] = SVR_pred_test_Y.flatten()*(self.Y_max - self.Y_min) + self.Y_min
        color_list = [  "b", "#02B2FF", "#02FFFF", "#28FF02", "y", "#FF9402", "#FF5602", "r", "#800000"]
        output_H = range(28, 37, 1)
        output_L = range(27, 36, 1)
        fig, ax = plt.subplots(figsize=(12,6))
        plot_x = "Air Temperature"
        plot_y = "Relative Humidity"
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(Rh_low_limit, Rh_high_limit)
        ax.set_ylabel("Relative Humidity (%)", fontsize=18)
        ax.set_xlabel('Operative Temperature ($^\circ$C)', fontsize=18)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(air_low_limit, air_high_limit + 4)
        ax.tick_params(labelsize = 18)
        plt.xticks(np.arange(17, air_high_limit, 1.0))
        
        ax.scatter(test_df[plot_x], test_df[plot_y],
             c=test_df["Skin Temperature"], cmap='jet')
        if self.simulation == True:
            plt.title(self.user + " Prediction with Random Temperature and Predicted Humidity" ,fontsize=20)
        else:
            plt.title("Prediction using 30%% real data for testing, Error: (SVR:%1.2f)" % (MSE_SVR), fontsize=20)

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
        plt.tight_layout()
        if self.simulation == True:
            plt.savefig(self.filename + "_" + plot_x + "_" + plot_y + "_SVR_simulation")
        else:
            plt.savefig(self.filename + "_" + plot_x + "_" + plot_y + "_SVR")
        plt.close()
        return    


    def analyze_data_Ta_Rh(self, data):
        #https://matplotlib.org/devdocs/gallery/text_labels_and_annotations/custom_legends.html
        color_list = [  "b", "#02B2FF", "#02FFFF", "#28FF02", "y", "#FF9402", "#FF5602", "r", "#800000"]
        data["Date"] = data.index.date
        output_H = range(28, 37, 1)
        output_L = range(27, 36, 1)
        fig, ax = plt.subplots(figsize=(12,6))
        plot_x = "Air Temperature"
        plot_y = "Relative Humidity"
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(Rh_low_limit, Rh_high_limit)
        ax.set_ylabel("Relative Humidity (%)", fontsize=18)
        ax.set_xlabel('Operative Temperature ($^\circ$C)', fontsize=18)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(air_low_limit, air_high_limit + 4)
        ax.tick_params(labelsize = 18)
        plt.xticks(np.arange(17, air_high_limit, 1.0))
        date_list = data["Date"].unique()
        marker_list = ["o", "*", "p", ">", "<", "d", "^", "x"]

        # for j in range(len(date_list)):
        #     ax.scatter(data.loc[(data["Date"] == date_list[j])][plot_x], 
        #             data.loc[(data["Date"] == date_list[j])][plot_y],
        #            c = data.loc[(data["Date"] == date_list[j])]["Skin Temperature"], 
        #            cmap='jet', marker=marker_list[j])
        for j in range(len(date_list)):
            for i in range(len(output_H)):
                if i == 0: # the first interval
                    ax.scatter(data.loc[(data["Skin Temperature"] <= output_L[1])
                    & (data["Date"] == date_list[j])][plot_x], 
                    data.loc[(data["Skin Temperature"] <= output_L[1])
                     & (data["Date"] == date_list[j])][plot_y],
                    color = color_list[i], marker=marker_list[j], label= str(date_list[j]))
                elif i == len(date_list) - 1: # the last interval
                    ax.scatter(data.loc[(data["Skin Temperature"] > output_H[i-1]) & (data["Date"] == date_list[j])][plot_x], 
                    data.loc[(data["Skin Temperature"] > output_H[i-1]) & (data["Date"] == date_list[j])][plot_y],
                    color = color_list[i], marker=marker_list[j], label= str(date_list[j]))
                else:
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
        lengend_list = [" Outdoor:(Ta$^\circ$C, Rh%)"]
        lengend_list = lengend_list + [str(date_list[i]) + ":" 
            + str(self.outdoor[i]) for i in range(len(date_list))]
        legend1 = plt.legend(custom_legend, lengend_list, loc='upper right',
         fontsize=12)  

        custom_legend = [Line2D([0], [0], marker= "o",  color='#E5E8E8', 
         markerfacecolor = color, markersize="8") for color in color_list]
        lengend_list = []
        lengend_list = ["Skin <= 28$^\circ$C", "28$^\circ$C < Skin <= 29$^\circ$C", 
        "29$^\circ$C < Skin <= 30$^\circ$C", "30$^\circ$C < Skin <= 31$^\circ$C", 
        "31$^\circ$C < Skin <= 32$^\circ$C", "32$^\circ$C < Skin <= 33$^\circ$C",
        "33$^\circ$C < Skin <= 34$^\circ$C", "34$^\circ$C < Skin <= 35$^\circ$C",
        "35$^\circ$C < Skin "]
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
                 Rh_low_limit, Rh_high_limit)
            plot_observation(one_data, self.location + "skin/" + "Temperature_Skin_" + tag, 
                "Air Temperature", "Skin Temperature", False, tag, air_low_limit, air_high_limit, 
                 skin_low_limit, skin_high_limit)
            i += 1
        plot_observation(data, self.location + "skin/" + "Temperature_Humidity_" + self.user, 
                 "Air Temperature", "Relative Humidity", False, self.user, air_low_limit, air_high_limit, 
                 Rh_low_limit, Rh_high_limit)
        plot_observation(data, self.location + "skin/" + "Temperature_Skin_" + self.user, 
                "Air Temperature", "Skin Temperature", False, self.user, air_low_limit, air_high_limit, 
                 skin_low_limit, skin_high_limit)
        plot_observation(data, self.location + "skin/" + "Humidity_Skin_" + self.user, 
                "Relative Humidity", "Skin Temperature", False, self.user,  Rh_low_limit, 
                Rh_high_limit, skin_low_limit, skin_high_limit)
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
        data_skin = data_skin.resample('60s').mean()

        # ####### process air temperature and relative humidity
        data_air = pd.read_csv(air_File, names = ["Time", "Air Temperature", "Relative Humidity"])
        data_air.index = pd.to_datetime(data_air.Time)
        data_air = data_air.resample('60s').mean().bfill()
        ###### Merge Skin and Environmental Data and drop wrong data
        training_set = pd.merge(data_skin, data_air, how='inner', left_index=True, right_index=True)

        return training_set.dropna(), outdoor
    


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
    def __init__(self, user, output, tags,  simulate_train = False, further_train= False, simulation = False, figure=None, is_3_scale=False): 
        self.simulation = simulation
        self.simulate_train = simulate_train
        self.further_train = further_train
        self.location = "csv/" + user + "/"
        self.tags = tags
        self.user  = user
        self.output = output
        self.selected_features = {"user1": ["Skin_0m", "Temperature_0m", "Humidity_0m"], "user2": ["Skin_0m", "Temperature_0m", "Heart_0m"], 
                "user3": ["Skin_0m", "Temperature_0m", "Humidity_0m"], "user4": ["Skin_0m", "Temperature_0m", "Humidity_0m"], 
                "user5": ["Skin_0m", "Temperature_0m", "Out_Humidity_0m"],"user6": ["Skin_0m", "Temperature_0m", "Out_Temperature_0m"]}
        self.selected_features_high = {"user1": [skin_high_limit, air_high_limit, Rh_high_limit], 
        "user2": [skin_high_limit, air_high_limit, hrate_high_limit], 
        "user3": [skin_high_limit, air_high_limit, Rh_high_limit], "user4": [skin_high_limit, air_high_limit, Rh_high_limit], 
        "user5": [skin_high_limit, air_high_limit, Rh_out_high_limit],"user6": [skin_high_limit, air_high_limit, Ta_out_high_limit]}
        self.selected_features_low= {"user1": [skin_low_limit, air_low_limit, Rh_low_limit], 
        "user2": [skin_low_limit, air_low_limit, hrate_low_limit], 
        "user3": [skin_low_limit, air_low_limit, Rh_low_limit], "user4": [skin_low_limit, air_low_limit, Rh_low_limit], 
        "user5": [skin_low_limit, air_low_limit, Rh_out_low_limit], "user6": [skin_low_limit, air_low_limit, Ta_out_low_limit]}
        self.input =  ["Skin_0m", "Temperature_0m", "Humidity_0m"]#self.selected_features[self.user] #["Skin_0m", "Temperature_0m", "Humidity_0m", "Out_Temperature_0m",  "Out_Humidity_0m", "Heart_0m","RR_0m",  "Skin_Delta_5m", "Skin_Delta_3m",  "Air_Delta_5m", "Air_Delta_3m"]#self.selected_features[self.user] 
        self.pred_Y = None
        self.accuracy  = 0
        self.test_acu = 0
        self.X_max =  np.asarray([skin_high_limit, air_high_limit, Rh_high_limit])#np.asarray(self.selected_features_high[self.user]) #np.asarray([skin_high_limit, air_high_limit, Rh_high_limit, Ta_out_high_limit, Rh_out_high_limit, hrate_high_limit, rr_high_limit, skin5m_high_limit, skin3m_high_limit, Air5m_high_limit, Air3m_high_limit ]) #self.selected_features_high[self.user
        self.X_min =  np.asarray([skin_low_limit, air_low_limit, Rh_low_limit])#np.asarray(self.selected_features_low[self.user]) # np.asarray([skin_low_limit, air_low_limit, hrate_low_limit, Ta_out_low_limit, Rh_out_low_limit, hrate_low_limit, rr_low_limit, skin5m_low_limit, skin3m_low_limit, Air5m_low_limit, Air3m_low_limit]) # self.selected_features_low[self.user]
        if simulate_train == True:
            self.filename = self.location + "subjective/" + output + "_" + user + "_" + "PMV"
            skin, Ta, humidity = self.simulate_input_data(1000, 0.1)
            self.simulatedData = {
                "Skin_0m": skin,
                "Temperature_0m": Ta,
                "Humidity_0m": humidity,
                }
            self.data = pd.DataFrame(self.simulatedData).ix[:,["Skin_0m", "Temperature_0m", "Humidity_0m"]] #
            satisfaction, sensation = self.PMV(self.data)
            self.data["Satisfaction"] = satisfaction
            self.data["Sensation"] =sensation
        else: 
            if further_train == True:
                self.filename = self.location + "subjective/" + output + "_" + user + "_" + "PMV_Real"
            else:
                self.filename = self.location + "subjective/" + output + "_" + user
            data_with_nan, self.outdoor = self.combine_data()

             ## to get last 5-minutes data will get more nan
            self.regular_data = data_with_nan[self.input + [output]]
            self.regular_data  = self.regular_data.dropna()
            self.data = data_with_nan.dropna()
            self.data["Skin_Delta_5m"] = self.data["Skin_0m"].subtract(self.data["Skin_5m"]).abs()
            self.data["Skin_Delta_3m"] = self.data["Skin_0m"].subtract(self.data["Skin_3m"]).abs()
            self.data["Air_Delta_5m"] = self.data["Temperature_0m"].subtract(self.data["Temperature_5m"]).abs()
            self.data["Air_Delta_3m"] = self.data["Temperature_0m"].subtract(self.data["Temperature_3m"]).abs()

        self.X = self.regular_data.ix[:, self.input].as_matrix()

        # scaler = MinMaxScaler()
        # scaler.fit(self.data.ix[:, self.input])
        # print(scaler.data_max_)
        # print(scaler.data_min_)
        #self.X = self.data.ix[:, ["Skin_0m"]].as_matrix().reshape(-1, 1)
        seed(1)
        X_scaled = (self.X - self.X_min)/(self.X_max - self.X_min)
        ##### encode class values as integers
       
        if output == "Satisfaction":
             # add -3 and 3 at front to get 7 lable
            self.Y = np.array(np.append(self.regular_data[output].as_matrix(), [-3, 3])).reshape(-1, 1)
        else:
            #self.Y = (np.array(self.data[output].as_matrix()) + 2).reshape(-1, 1)
            #self.Y = np.array(np.append(self.data[output].as_matrix(), [-3, 3])).reshape(-1, 1)
            # # group output to three labels:
            self.regular_data.loc[self.regular_data[output] < 0, output] = -1
            self.regular_data.loc[self.regular_data[output] > 0, output] = 1
            self.Y = np.array(self.regular_data[output].as_matrix()).reshape(-1, 1)
        

        encoder = LabelEncoder()
        encoder.fit(self.Y)
        encoded_Y = encoder.transform(self.Y)
        ### convert integers to dummy variables (i.e. one hot encoded)
        # if output == "Satisfaction":
        #     #remove the appended fake data
        #     dummy_y = np_utils.to_categorical(encoded_Y)[:-2]
        # else:
        #     dummy_y = np_utils.to_categorical(encoded_Y)[:-2]
        dummy_y = np_utils.to_categorical(encoded_Y)
        self.lable_num = dummy_y.shape[1]
        # for neural network
        # self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(
        #     X_scaled, dummy_y, test_size=0.4, random_state=42)
        # for other algroithem 
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(
            X_scaled, self.Y, test_size=0.4, random_state=42)
        if simulate_train == False:
            if self.simulation == False: 
                if further_train == False:
                    #self.analyze_data_Ta_Rh_Skin()
                    # self.analyze_data_Ta_Rh(data_with_nan)
                    # if self.output == "Sensation":
                    #     ##used for combined users data:  
                    #     #self.analyze_distribution(self.data, figure)
                          #self.analyze_three_distribution(self.data)
                    #     # self.analyze_data_scatter(data_with_nan,"Temperature_", air_low_limit, air_high_limit)
                    #     # self.analyze_data_scatter(data_with_nan, "Skin_" ,skin_low_limit, skin_high_limit)
                    #     # self.analze_data_violin(data_with_nan, "Temperature_0m", air_low_limit, air_high_limit)
                    #     #self.analze_data_violin(data_with_nan, "Skin_0m", skin_low_limit, skin_high_limit)
                    #     # self.analze_data_violin(data_with_nan, "Temperature_5m", 0, 1.8)
                    #     #self.analze_data_violin(data_with_nan, "Skin_5m", 0, 1.1)
                    #     # self.analze_data_violin(data_with_nan, "Temperature_3m", 0, 1.1)
                    #     #self.analze_data_violin(data_with_nan, "Skin_3m", 0, 0.7)
                    # else:
                    #     ##used for combined users data: 
                    #     # self.analyze_distribution(self.data, figure)
                    #     # return
                    #     self.analyze_data_scatter_h(data_with_nan,"Temperature_", air_low_limit, air_high_limit)
                    #     self.analyze_data_scatter_h(data_with_nan, "Skin_" , skin_low_limit, skin_high_limit)
                    #     self.analze_data_violin(data_with_nan, "Skin_0m", skin_low_limit, skin_high_limit)
                    #     self.analze_data_violin(data_with_nan, "Temperature_5m", 0, 1.8)
                    #     self.analze_data_violin(data_with_nan, "Skin_5m", 0, 1.1)
                    #     self.analze_data_violin(data_with_nan, "Temperature_3m", 0, 1.1)
                    #     self.analze_data_violin(data_with_nan, "Skin_3m", 0, 1.0)

                    self.analyze_data_delta(self.data, 5, "Temperature_")
                    self.analyze_data_delta(self.data, 5, "Skin_")
            else:
                skin, Ta, humidity = self.simulate_input_data(400, 0.1)
                self.simulatedData = {
                "Skin": (skin - skin_low_limit)/(skin_high_limit - skin_low_limit),
                "Ta": (Ta - air_low_limit)/(air_high_limit - air_low_limit),
                "Rh": (humidity - Rh_low_limit)/(Rh_high_limit - Rh_low_limit),
                # "Out_Ta": (np.random.choice(np.arange(8, 9, 0.1),
                #          1000) - Ta_out_low_limit)/(Ta_out_high_limit - Ta_out_low_limit),
                # "Out_Rh": (np.random.choice(np.arange(30, 31, 0.1),
                #          1000) - Rh_out_low_limit)/(Rh_out_high_limit - Rh_out_low_limit)
                }
                self.simulatedDf = pd.DataFrame(self.simulatedData).ix[:,["Skin", "Ta", "Rh"]] #,"Out_Ta", "Out_Rh"
                self.test_X = self.simulatedDf.as_matrix()    
          
            
    
    def simulate_input_data(self, num_data, data_interval):
        # predict humidity based on random selected indoor tempeature and outdoor humidity in certrain ranges 
        model_Rh = pickle.load(open(self.location + "environment/temperature_" + self.user + "_RH_kernel.sav", 'rb'))
        Ta = (np.random.choice(np.arange(air_low_limit, air_high_limit, data_interval), num_data) 
            - air_low_limit)/(air_high_limit - air_low_limit)
        Rh_out = (np.random.choice(np.arange(Rh_out_low_limit, Rh_out_high_limit, data_interval),
            num_data) - Rh_out_low_limit)/(Rh_out_high_limit - Rh_out_low_limit)
        self.simulatedData1 = {
        "Ta": Ta,
        "Rh_out": Rh_out
        }
        self.simulatedDf1 = pd.DataFrame(self.simulatedData1).ix[:,["Ta","Rh_out"]]
        self.test_X1= self.simulatedDf1.as_matrix()
        humidity = model_Rh.predict(self.test_X1).flatten()
        # combine random generated indoor temperature and predicted humidity for skin temperature prediciton
        self.simulatedData2 = {
        "Ta": Ta,
        "Rh": humidity,
        }
        self.simulatedDf2 = pd.DataFrame(self.simulatedData2).ix[:,["Ta", "Rh"]]
        self.test_X2 = self.simulatedDf2.as_matrix()
        model_Skin = pickle.load(open(self.location + "skin/Skin_" + self.user + "_Skin Temperature_SVR.sav", 'rb'))
        skin = model_Skin.predict(self.test_X2).flatten()
        input1 = skin*(skin_high_limit - skin_low_limit) + skin_low_limit
        input2 = Ta*(air_high_limit - air_low_limit) + air_low_limit
        input3 = humidity*(Rh_high_limit - Rh_low_limit) + Rh_low_limit
        return (input1, input2, input3)
      


    def randomForest(self):
        #http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py
        
        #https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        # param_grid = {'bootstrap': [True, False],
        #  'max_depth': [2, 10, 20, 30, 40, 50, 60, None],
        #  'min_samples_leaf': [1, 2, 4],
        #  'min_samples_split': [2, 5, 10],
        #  'n_estimators': [2, 10, 50, 100, 200, 400],
        #  }
        # clf = RandomForestClassifier(random_state=0)
        # grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
        #                   cv = 3, n_jobs = -1)
        # c, r = self.train_Y.shape
        # labels = self.train_Y.reshape(c,)
        # grid_search.fit(self.train_X, labels)
        # best_grid = grid_search.best_estimator_
        # self.pred_Y = best_grid.predict(self.test_X)
        # accuracy = accuracy_score(self.pred_Y, self.test_Y)
        # with open("csv/forest_"  + self.user + ".txt", 'a') as the_file:
        #     the_file.write('Best parameters set found on development set:\n')
        #     the_file.write(json.dumps(grid_search.best_params_))
        #     the_file.write('\n')
        #     the_file.write("Grid scores on development set:\n")
        #     means = grid_search.cv_results_['mean_test_score']
        #     stds = grid_search.cv_results_['std_test_score']
        #     for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        #         the_file.write("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        #         the_file.write('\n')
        #     the_file.write('Final Testing Accuracy: \n')
        #     the_file.write(str(accuracy))


       #http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
       #http://scikit-learn.org/stable/modules/feature_selection.html
        bootstrap = {"user1": True, "user2": False, "user3": True, "user4": True, "user5": False, "user6": True}
        max_depth = {"user1": 10, "user2": 10, "user3": 10, "user4": 10, "user5": 10, "user6": 10}
        max_features = {"user1": "auto", "user2": "auto", "user3": "auto", "user4": "auto", "user5": "auto", "user6": "auto"}
        min_samples_leaf = {"user1": 1, "user2": 1, "user3": 1, "user4": 1, "user5": 2, "user6": 1}
        min_samples_split = {"user1": 5, "user2": 2, "user3": 5, "user4": 2, "user5": 2, "user6": 5}
        n_estimators = {"user1": 400, "user2": 200, "user3": 50, "user4": 200, "user5": 10, "user6": 10}
        class_weights = {"user1": None, "user2": 'balanced', "user3": 'balanced', "user4": None, "user5": None, "user6": None}
        forest = RandomForestClassifier(class_weight=class_weights[self.user], bootstrap = bootstrap[self.user], max_depth = max_depth[self.user], 
            max_features =  max_features[self.user], min_samples_leaf =  min_samples_leaf[self.user], 
            min_samples_split =  min_samples_split[self.user], n_estimators =  n_estimators[self.user],
            random_state=0)

        if(self.simulation == False):
            forest.fit(self.train_X, self.train_Y)
            self.pred_Y = forest.predict(self.test_X)
            self.accuracy = accuracy_score(self.pred_Y, self.test_Y)
            # save the model to disk
            filename = 'finalized_model.sav'
            pickle.dump(forest, open("csv/randomForest/" + self.user, 'wb'))
            print(self.accuracy)
        else:
            # load the model from disk
            forest = pickle.load(open("csv/randomForest/" + self.user, 'rb'))
            self.pred_Y = forest.predict(self.test_X)
            print(self.pred_Y)
            self.cacluation_T_setpoint(forest)

        # importances = forest.feature_importances_
        # std = np.std([tree.feature_importances_ for tree in forest.estimators_],
        #      axis=0)
        # indices = np.argsort(importances)[::-1]

        # # Print the feature ranking
        # print("Feature ranking:")

        # for f in range(self.train_X.shape[1]):
        #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # # Plot the feature importances of the forest
        # plt.figure()
        # plt.title("Feature importances")
        # plt.bar(range(self.train_X.shape[1]), importances[indices],
        #        color="r", yerr=std[indices], align="center")
        # plt.xticks(range(self.train_X.shape[1]), indices)
        # plt.xlim([-1, self.train_X.shape[1]])
        # plt.show()
 


       
        

    def svm(self):
       # Grid Search CV to find the best parameters for each users
        c, r = self.train_Y.shape
        self.train_Y = self.train_Y.reshape(c,)

        param_grid = [
          {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'class_weight' : ['balanced', None]},
          {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001, 'auto'], 'kernel': ['rbf'], 'class_weight' : ['balanced', None]},
         ]
        svc = svm.SVC(random_state=42)
        grid_search = GridSearchCV(estimator = svc, param_grid = param_grid, 
                          cv = 3)
        grid_search.fit(self.train_X, self.train_Y) 
        print(grid_search.best_params_)
        best_svc = grid_search.best_estimator_
        self.pred_Y = best_svc.predict(self.test_X)
        accuracy = accuracy_score(self.pred_Y, self.test_Y)
        print(accuracy)


        # # using the best parameters to train and test
        # parameters for only air tempeature, relative humidity
        #best_parameters = {"user1": {"kernel": "linear", 'C':1} , "user2": {"kernel": "linear", 'C':10} , "user3": {"kernel": "linear", 'C':10}, "user4": {"kernel": "linear", 'C':100}, "user5": {"kernel": "linear", 'C':10}, "user6": {"kernel": "linear", 'C':1000}}
        # parameters for skin temperature, air tempeature, relative humidity
        # best_parameters = {"user1": {"kernel": "linear", 'C':100} , "user2": {"kernel": "linear", 'C':1} , "user3": {"kernel": "linear", 'C':1000}, "user4": {"kernel": "linear", 'C':100}, "user5": {"kernel": "linear", 'C':100}, "user6": {"kernel": "linear", 'C':1000}}
        # clf = svm.SVC(kernel = best_parameters[self.user]["kernel"], C = best_parameters[self.user]["C"])
        # clf.fit(self.train_X, self.train_Y) 
        # self.pred_Y = clf.predict(self.test_X)
        # accuracy = accuracy_score(self.pred_Y, self.test_Y)
        # print("Testing Accuracy:" + str(accuracy))
       



    def neural_network(self, optimizer):
        # ref:https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/ 
        ## optimizer options: adam, sgd


        #https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
        
        model = Sequential()
        model.add(Dense(len(self.input), input_dim=len(self.input), activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(3, activation='softmax'))  #8,6,3
        # Compile model
        model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, 
                  metrics=['accuracy'])
        
        if self.simulation == True:
            if self.further_train == True:
                model.load_weights(self.location  + 'subjective/' + self.user + '_PMV_Real_' + self.output + '_ANN.h5')
            else:
                model.load_weights(self.location  + 'subjective/'  + self.user + '_' + self.output + '_ANN.h5')
        else:
            if self.further_train == True:
                model.load_weights(self.location  + 'subjective/'  + self.user + '_PMV_' + self.output + '_ANN.h5')
                loss_and_metrics = model.evaluate(self.test_X,  self.test_Y, batch_size=2)  
                self.test_acu = loss_and_metrics[1]
            model.fit(self.train_X, self.train_Y, epochs=200, batch_size=2, verbose=0)
            if self.simulate_train == True:
                model.save_weights(self.location + "subjective/" + self.user + "_PMV_"  + self.output + "_ANN.h5")
            elif self.further_train == True:
                model.save_weights(self.location + "subjective/" + self.user + "_PMV_Real_"  + self.output + "_ANN.h5")
            else: 
                model.save_weights(self.location  + 'subjective/'  + self.user + '_' + self.output + '_ANN.h5')
            loss_and_metrics = model.evaluate(self.test_X,  self.test_Y, batch_size=2)[1]
            print(loss_and_metrics)
            self.test_Y = [np.argmax(values) + min(self.Y.flatten()) for values in self.test_Y ]
        classes = model.predict(self.test_X, batch_size=2)      
        self.pred_Y = [np.argmax(values) + min(self.Y.flatten()) for values in classes]
        self.accuracy = accuracy_score(self.pred_Y, self.test_Y)
        print(self.accuracy)


    def cacluation_T_setpoint(self, model):
        data_interval = 0.1
        data_num = 1000
        skin = np.random.choice(np.arange(skin_low_limit, skin_high_limit, data_interval), data_num) 
        Ta = np.random.choice(np.arange(air_low_limit, air_high_limit, data_interval), data_num)
        Rh = np.asarray([30]*data_num)
        simulatedData = {
            "Skin": (skin - skin_low_limit)/(skin_high_limit - skin_low_limit),
            "Ta": (Ta - air_low_limit)/(air_high_limit - air_low_limit),
            "Rh": (Rh - Rh_low_limit)/(Rh_high_limit - Rh_low_limit),
            }
        simulatedDf = pd.DataFrame(simulatedData).ix[:,["Skin", "Ta", "Rh"]] 
        test_X = simulatedDf.as_matrix()
        pred_Y = model.predict(test_X)
        test_X = test_X*(self.X_max - self.X_min) + self.X_min
        test_df = pd.DataFrame(test_X, columns = ["Skin", "Ta", "Rh"])
        data = test_df
        data["label"] = pred_Y
        final_data = data.loc[(data["label"] == 0)]
        
        print(final_data.loc[(data["Skin"] == 25)])
        for i in range(6):
            fig, ax = plt.subplots(figsize=(12,6))

            ax.plot(final_data["Skin"], final_data["Ta"],
             "b.", markersize="9", label='Observed')
            ax.tick_params(labelsize = 18)  
            plt.tight_layout() 
            plt.savefig("csv/randomForest/T_setpoint")
            plt.close()







    def PMV(self):
        satisfaction = []
        sensation = []
        test_X = self.test_X*(self.X_max - self.X_min) + self.X_min
        test_df = pd.DataFrame(test_X, columns = self.input)
        for index, row in test_df.iterrows():
            r1, r2 = self.comfPMV(row["Temperature_0m"], row["Temperature_0m"], row["Humidity_0m"]) 
            satisfaction.append(r1)   
            sensation.append(r2)   
        sen = np.array(sensation)
        np.putmask(sen, sen>1, [1])
        np.putmask(sen, sen<-1, [-1])
        accuracy_PMV = accuracy_score(sen, self.test_Y)
        count = 0
        # for i in range(len(sensation)):
        #     if sensation[i] == self.test_Y[i]:
        #         count += 1
        #         accuracy_PMV = count*1.0/len(sensation)
        print(accuracy_PMV)
        return satisfaction, sensation


    def evaluation(self, method_name, is_3D):
        if self.simulation == False:
            fig, ax = plt.subplots(figsize=(12,8))

            ax.set(title=method_name + ":(accuracy:%f;)" % (self.accuracy ), xlabel='Time', ylabel=self.output)

            # Plot data points
            pd.DataFrame({"obervation": self.test_Y.flatten()}).plot(ax=ax, style='b.', markersize=10, label='Observed')

            # Plot predictions with time
            pd.DataFrame({"prediction": self.pred_Y}).plot(ax=ax, style='r.') 
            legend = ax.legend(loc='lower right')  
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(-3.2, 3.2)       
            ax.tick_params(labelsize = 18)                                         
            plt.savefig(self.filename + "_" + method_name)
            plt.close()


        test_X = self.test_X*(self.X_max - self.X_min) + self.X_min
        test_df = pd.DataFrame(test_X, columns = self.input)
        if(self.further_train == True and self.simulation == False):
            # put real testing data into PMV equation
            satisfaction, sensation = self.PMV()
            count = 0 
            if self.output == "Satisfaction":
                for i in range(len(satisfaction)):
                    if satisfaction[i] == self.test_Y[i]:
                        count += 1
                accuracy_PMV = count*1.0/len(satisfaction)
            else:
                for i in range(len(sensation)):
                    if sensation[i] == self.test_Y[i]:
                        count += 1
                accuracy_PMV = count*1.0/len(sensation)
               
        if is_3D == True:
            if (self.simulation == True):
                self.regular_data = test_df
                self.regular_data["Sensation"] = self.pred_Y
                self.analyze_data_Ta_Rh_Skin(method_name=method_name)
        else:
            plot_x = "Temperature_0m"
            plot_y = "Humidity_0m"
            test_df["Prediction"] = self.pred_Y
            if(self.output == "Satisfaction"):
                color_list = ["#800000", "r", "#ff8000", "#66ccff", "#33ff33", "#00cc00", "#006600"]
            else:
                color_list = [ "b", "#33ffff", "#3399ff", "#33ff66", "#ff8000", "#ff4000", "#800000"]
            output = [-3, -2,-1, 0, 1, 2, 3]
            fig, ax = plt.subplots(figsize=(12,6))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(Rh_low_limit, Rh_high_limit)
            ax.set_ylabel("Relative Humidity (%)",  fontsize=18)
            ax.set_xlabel('Operative Temperature ($^\circ$C)', fontsize=18)
            #ax.set_xlim(18, 31.2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xticks(np.arange(16, air_high_limit + 1, 1.0))
            for i in range(len(output)):
                ax.scatter(test_df.loc[test_df["Prediction"] == output[i]][plot_x], 
                    test_df.loc[test_df["Prediction"] == output[i]][plot_y],
                 color = color_list[i], label= self.output + ":" + str(output[i]))
            if self.simulation == False:
                if self.further_train == True:
                    plt.title(self.user + ": PMV+Real Neutral Network:%1.2f, PMV Neutral Network:%1.2f,PMV :%1.2f" % (
                    self.accuracy , self.test_acu, accuracy_PMV), fontsize=20)
                else:
                    plt.title(self.user + ": Predicted Thermal Comfort, accuracy:%1.2f" % (
                    self.accuracy ), fontsize=20)
            else:
                if self.further_train == True:
                    plt.title(self.user + ": PMV Pretrained Neutral Network with Simulated Input", fontsize=20)
                else:
                    plt.title(self.user + ": Field Personal Data "  + method_name +" with Simulated Input", fontsize=20)

            custom_legend = [Line2D([0], [0], marker= "o",  color='#E5E8E8', 
             markerfacecolor = color, markersize="8") for color in color_list]
            ax.tick_params(labelsize = 18)
            lengend_list = []
            if self.output == "Satisfaction": 
                lengend_list = ["Strongly Dissatisfied", "Dissatisfied", 
                "Slightly Dissatisfied", "Neutral", "Slightly Satisfied", 
                "Satisfied", "Strongly Satisfied"]
            else: 
                lengend_list = ["Cold", "Cool", 
                "Slightly Cool", "Neutral", "Slightly Warm", 
                "Warm", "Hot"]
            plt.legend(custom_legend, lengend_list , loc='upper right',
             fontsize=12) 
            plt.tight_layout() 
            if self.simulation == False: 
                plt.savefig(self.filename + "_" + plot_x + "_" + plot_y + "_" + method_name)
            else:
                plt.savefig(self.filename + "_" + plot_x + "_" + plot_y + "_" + method_name + "_simulation" )
            plt.close()  


    def comfPMV(self, ta, tr, rh,  clo=1.0, vel=0.1, met =1.0, wme = 0):
        """
        ref:https://github.com/CenterForTheBuiltEnvironment/comfort_tool/blob/master/contrib/comfort_models.py
        returns [pmv, ppd]
        ta, air temperature (C), must be float, e.g. 25.0
        tr, mean radiant temperature (C),must be float, e.g. 25.0
        vel, relative air velocity (m/s), must be float, e.g. 25.0
        rh, relative humidity (%) Used only this way to input humidity level
        met, metabolic rate (met)
        clo, clothing (clo)
        wme, external work, normally around 0 (met)
        """

        pa = rh * 10 * math.exp(16.6536 - 4030.183 / (ta + 235))

        icl = 0.155 * clo  # thermal insulation of the clothing in M2K/W
        m = met * 58.15  # metabolic rate in W/M2
        w = wme * 58.15  # external work in W/M2
        mw = m - w  # internal heat production in the human body
        if (icl <= 0.078):
            fcl = 1 + (1.29 * icl)
        else:
            fcl = 1.05 + (0.645 * icl)

        # heat transf. coeff. by forced convection
        hcf = 12.1 * math.sqrt(vel)
        taa = ta + 273
        tra = tr + 273
        tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)

        p1 = icl * fcl
        p2 = p1 * 3.96
        p3 = p1 * 100
        p4 = p1 * taa
        p5 = (308.7 - 0.028 * mw) + (p2 * math.pow(tra / 100, 4))
        xn = tcla / 100
        xf = tcla / 50
        eps = 0.00015

        n = 0
        while abs(xn - xf) > eps:
            xf = (xf + xn) / 2
            hcn = 2.38 * math.pow(abs(100.0 * xf - taa), 0.25)
            if (hcf > hcn):
                hc = hcf
            else:
                hc = hcn
            xn = (p5 + p4 * hc - p2 * math.pow(xf, 4)) / (100 + p3 * hc)
            n += 1
            if (n > 150):
                print ('Max iterations exceeded')
                return 1


        tcl = 100 * xn - 273

        # heat loss diff. through skin
        hl1 = 3.05 * 0.001 * (5733 - (6.99 * mw) - pa)
        # heat loss by sweating
        if mw > 58.15:
            hl2 = 0.42 * (mw - 58.15)
        else:
            hl2 = 0
        # latent respiration heat loss
        hl3 = 1.7 * 0.00001 * m * (5867 - pa)
        # dry respiration heat loss
        hl4 = 0.0014 * m * (34 - ta)
        # heat loss by radiation
        hl5 = 3.96 * fcl * (math.pow(xn, 4) - math.pow(tra / 100, 4))
        # heat loss by convection
        hl6 = fcl * hc * (tcl - ta)

        ts = 0.303 * math.exp(-0.036 * m) + 0.028
        pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
        ppd = 100.0 - 95.0 * math.exp(-0.03353 * pow(pmv, 4.0)
            - 0.2179 * pow(pmv, 2.0))

        if ppd < 8: # pmv in (-0.3, 0.3)
            satisfaction = 3
        elif ppd < 20: # pmv in (-0.8, 0.8)
            satisfaction = 2
        elif ppd < 35: # pmv in (-1.3, 1.3)
            satisfaction = 1
        elif ppd < 50: # pmv in (-1.5, 1.5)
            satisfaction = 0
        elif ppd < 65: # pmv in (-1.8, 1.8)
            satisfaction = -1
        elif ppd < 80: # pmv in (-2.1, 2.1)
            satisfaction = -2
        else: # ppd > 0.8
            satisfaction = -3 

        if pmv < 0.5 and pmv > -0.5:
            sensation = 0
        elif pmv > 0.5  and pmv < 1.5:
            sensation = 1
        elif pmv > 1.5 and pmv < 2.5:
            sensation = 2
        elif pmv > 2.5:
            sensation = 3
        elif pmv > -1.5 and pmv < -0.5:
            sensation = -1
        elif pmv > -2.5 and pmv < -1.5:
            sensation = -2
        else: 
            sensation = -3
        return satisfaction, sensation

    def mean_confidence_interval(self, data, confidence=0.95):
        a = 1.0*np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
        return m, h


    def analyze_three_distribution(self, data):
        fig, ax = plt.subplots(figsize=(12,6))
        input_ = "Skin"
        if self.output == "Satisfaction":
            comfort_data = data.loc[data[self.output] > 0]
            title = input_ + " Temperature Gaussian Distribution for Satisfaction > 0"
        else:
            comfort_data = data.loc[(data[self.output] == 0)]["Skin_0m"].tolist()
            print(statistics.median(comfort_data))
            print(statistics.mean(comfort_data))
            hot =  data.loc[(data[self.output] > 0)]["Skin_0m"].tolist()
            cold =  data.loc[(data[self.output] < 0)]["Skin_0m"].tolist()
            title = self.user 
 
        # Fit a normal distribution to the data:
        mu_comfort, std_comfort = norm.fit(comfort_data)
       
        x = np.linspace(skin_low_limit + 1, skin_high_limit, 200)
    
        #mu_cold, std_cold = halfnorm.fit(cold)
        plt.hist(cold, bins=10, normed=True, alpha=0.1, color='b')
        # Plot the PDF.
       # p = halfnorm.pdf(x, mu_cold, std_cold)
        #ax.plot(x, p, 'b', linewidth=2, label="Cool(<0), n=" + str(len(cold)))

         #Plot the histogram.
        plt.hist(comfort_data, bins=20, normed=True, alpha=0.1, color='g')
        # Plot the PDF.
        #p = norm.pdf(x, mu_comfort, std_comfort)
        #ax.plot(x, p, 'g', linewidth=2, label="Comfort(0), n=" + str(len(comfort_data)))


        #mu_hot, std_hot = halfnorm.fit([-i for i in hot])

        plt.hist(hot, bins=20, normed=True, alpha=0.1, color='r')
         # Plot the PDF.
        #p = halfnorm.pdf([-i for i in x], mu_hot, std_hot)
        # x, y = np.mgrid[-1:1:.01, -1:1:.01]
        # pos = np.empty(x.shape + (2,))
        # pos[:, :, 0] = x; pos[:, :, 1] = y
        # rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
        # plt.contourf(x, y, rv.pdf(pos))
        # plt.show()

        #ax.plot(x, p, 'r', linewidth=2, label="Warm(>0), n=" + str(len(hot)))
        ax.set_xlim(26, 36)
        ax.set_ylim(0, 0.8)
        plt.legend()
        #plt.tight_layout()
        ax.set_ylabel("Probability")
        ax.set_xlabel(input_ + " Temperature ($^\circ$C)")
        #plt.title(title)
        plt.savefig("csv/" + self.user + "_skin_sensation")
        # accurate = 0
        # data = [cold, comfort_data, hot]
        # for i in range(3):
        #     density_list = np.transpose([halfnorm.pdf(data[i], mu_cold, std_cold), 
        #                 norm.pdf(data[i], mu_comfort, std_comfort), 
        #                 halfnorm.pdf(data[i], mu_hot, std_hot)])
        #     for item in density_list:
        #         if np.argmax(item) == i:
        #             accurate += 1
        # print(self.user + ":" + str(accurate*1.0/(len(cold) + len(comfort_data) + len(hot))))



    def analyze_distribution(self, data, figure):
        import matplotlib as mpl
        mpl.style.use("default")
        input_ = "Skin"
        if self.output == "Satisfaction":
            comfort_data = data.loc[data[self.output] > 0]
            title = input_ + " Temperature Gaussian Distribution for Satisfaction > 0"
        else:
            comfort_data = data.loc[(data[self.output] == 0 )]
            title = input_ + " Temperature Gaussian Distribution for Neutral Sensation"
        ax = figure[0]
        color = figure[1]
       
        comfort_data = comfort_data["Humidity_0m"].tolist()
        # Fit a normal distribution to the data:
        mu, std = halfnorm.fit(comfort_data)
        print(self.user)
        print(mu, std)

        # Plot the histogram.
        plt.hist(comfort_data, bins=30, normed=True, alpha=0.1, color=color)

        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(21, 40, 100)
        p = halfnorm.pdf(x, mu, std)
        ax.plot(x, p, color, linewidth=2, label=self.user) #'gold', 'olive', 'coral', 'C4', 'sienna', 'teal', 'tomato'


        # if(self.user == "user4"):
        #     x = np.linspace(xmin, xmax, 200)
        #     p = norm.pdf(x, 24.2, 2.5)
        #     ax.plot(x, p, "#000080", linewidth=2, label="All three users")
        ax.set_ylim(0, 0.3)
        plt.legend()
        plt.tight_layout()
        ax.set_ylabel("Probability")
        ax.set_xlabel(input_ + " Temperature($^\circ$C)")
        ax.set_title(title)




    def analze_data_violin(self, data, input_, limit_low, limit_high):
        print(self.filename)
        output = [ -3, -2,-1, 0, 1, 2, 3]
        fig, ax = plt.subplots(figsize=(12,6))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(0, 8)
        ax.set_xlabel('Thermal ' + self.output, fontsize=18)
        if(self.output == "Satisfaction"):
            color_list = ["#800000", "r", "#ff8000", "#66ccff", "#33ff33", "#00cc00", "#006600"]
        else:
            color_list = [ "b", "#33ffff", "#3399ff", "#33ff66", "#ff8000", "#ff4000", "#D43F3A"]
        if input_ == "Temperature_0m": 
            plot_x = input_
            text = "Air Temperature 1m Average($^\circ$C)"
        elif input_ == "Skin_0m":
            text = "Skin Temperature 1m Average($^\circ$C)"
            plot_x = input_
        elif input_ == "Temperature_5m": # show absolute delta
            text = "Air Temperature 5m Absolute Gradient($^\circ$C)"
            data["Delta"] = data["Temperature_0m"].subtract(data["Temperature_5m"]).abs()
            plot_x = "Delta"
        elif input_ == "Skin_5m": # show absolute delta
            text = "Skin Temperature 5m Absolute Gradient($^\circ$C)"
            data["Delta"] = data["Skin_0m"].subtract(data["Skin_5m"]).abs()
            plot_x = "Delta"
        elif input_ == "Temperature_3m": # show absolute delta
            text = "Air Temperature 3m Absolute Gradient($^\circ$C)"
            data["Delta"] = data["Temperature_0m"].subtract(data["Temperature_3m"]).abs()
            plot_x = "Delta"
        elif input_ == "Skin_3m": # show absolute delta
            text = "Skin Temperature 3m Absolute Gradient($^\circ$C)"
            data["Delta"] = data["Skin_0m"].subtract(data["Skin_3m"]).abs()
            plot_x = "Delta"
        ax.set_ylabel(text ,fontsize=18)
        ax.set_ylim(limit_low, limit_high)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        #plt.xticks(np.arange(26, 36.2, 1.0))
        all_data = []
        median = []
        mean = []
        errors = []
        for xlabel in output:
            data_x = data.dropna().loc[data.dropna()[self.output] == xlabel][plot_x].tolist()
            m, h =  self.mean_confidence_interval(data_x)
            mean.append(m)
            errors.append(h)
            if(len(data_x)  == 0):
                median.append(None)
                all_data.append([0])
            else:
                median.append(statistics.median(data_x))
                all_data.append(data_x)
        # sns.violinplot(data = all_data, ax= ax)
        #    # showmeans=False,
        #    # showmedians=True,  ax= ax)
        violin_parts = ax.violinplot(all_data, showmeans=False, showmedians=True, showextrema=False)
        j = 0
        for pc in violin_parts['bodies']:
            pc.set_color(color_list[j])
            pc.set_alpha(0.5)
            # pc.set_facecolor(color_list[j])
            # pc.set_edgecolor('black')
            j += 1
        ax.plot(range(1, 8), median, color='r')
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
        lengend_list = ["Mean and 0.95 Confidence Interval", "The trend of Median Skin"]
        plt.legend(custom_legend, lengend_list , loc='upper left', fontsize=14)   
        plt.tight_layout()
        #legend = ax.legend(loc='lower right')     
        plt.savefig(self.filename + "_" + input_ + "_violin")
        plt.close()


    def analyze_data_delta(self, data, interval, input_, date=""):
        data["Delta"] = data[input_+"0m"].subtract(data[input_ + str(interval) + "m"]).abs()
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
        ax.tick_params(labelsize = 18)
        plt.title(self.user + "_" + date + ": " + str(round(correlation[0], 2)), fontsize=20)  
        plt.tight_layout()  
        plt.savefig(self.filename + "_" + plot_x + "_" + input_ + "_" + str(interval) + "m" +  date)
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
            ax.tick_params(labelsize = 18)
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
            ax.tick_params(labelsize = 18)
            plt.legend(custom_legend, lengend_list , loc='upper left',
             fontsize=12)   
            plt.tight_layout()
            plt.savefig(self.filename + "_" + plot_x)
            plt.close()


    def analyze_data_Ta_Rh_Skin(self, date="", marker="o", method_name=""):
        #https://matplotlib.org/devdocs/gallery/text_labels_and_annotations/custom_legends.html
        if(self.output == "Satisfaction"):
            color_list = ["#800000", "r", "#ff8000", "#66ccff", "#33ff33", "#00cc00", "#006600"]
            output = [ -3, -2,-1, 0, 1, 2, 3]
        else:
            color_list = [ "b", "#33ff66", "r"]
            output = [-1, 0, 1]
        
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111, projection='3d')
        plot_x = "Temperature_0m"
        plot_y = "Humidity_0m"
        plot_z = "Skin_0m"
        ax.set_ylim(16, Rh_high_limit)
        ax.set_xlim(air_low_limit, air_high_limit)
        ax.set_zlim(skin_low_limit, 35)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.zaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("Relative Humidity (%)", fontsize=12)
        ax.set_xlabel('Air Temperature ($^\circ$C)', fontsize=12)
        ax.set_zlabel('Skin Temperature ($^\circ$C)', fontsize=12)
        ax.set_xticks(np.arange(17, air_high_limit, 1.0))
        ax.set_zticks(np.arange(skin_low_limit, 35, 1.0))

        print(self.user)


        for i in range(len(output)):
            ax.scatter(self.regular_data.loc[(self.regular_data[self.output] == output[i])][plot_x], 
                self.regular_data.loc[(self.regular_data[self.output] == output[i])][plot_y],
                self.regular_data.loc[(self.regular_data[self.output] == output[i])][plot_z],
             color = color_list[i])
        plt.title(self.user + ", Clo:" + clo[self.user] + ", Counts:" + str(len(self.regular_data.index)), fontsize=16)

        custom_legend = [Line2D([0], [0], marker= "o",  color='#E5E8E8', 
         markerfacecolor = color, markersize="8") for color in color_list]
        lengend_list = []
        if self.output == "Satisfaction": 
            lengend_list = ["Strongly Dissatisfied", "Dissatisfied", 
            "Slightly Dissatisfied", "Neutral", "Slightly Satisfied", 
            "Satisfied", "Strongly Satisfied"]
        else: 
            lengend_list = [ "Cool", "Neutral", "Warm"]
        ax.tick_params(labelsize = 12)
        plt.legend(custom_legend, lengend_list , loc='lower right',
         fontsize=12) 

        plt.tight_layout()

        plt.savefig("csv/3D/" + self.user+ "_" + plot_x + "_" + plot_y + "_" + method_name + date )
        plt.close()

    def analyze_data_Ta_Rh(self, data, date="", marker="o"):
        #https://matplotlib.org/devdocs/gallery/text_labels_and_annotations/custom_legends.html
        if(self.output == "Satisfaction"):
            color_list = ["#800000", "r", "#ff8000", "#66ccff", "#33ff33", "#00cc00", "#006600"]
            offset = 3
        else:
            color_list = [ "b", "#33ffff", "#3399ff", "#33ff66", "#ff8000", "#ff4000", "#800000"]
            offset = 2
        data["Date"] = data.index.date
        output = [ -3, -2,-1, 0, 1, 2, 3]
        fig, ax = plt.subplots(figsize=(12,6))
        plot_x = "Temperature_0m"
        plot_y = "Humidity_0m"
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(Rh_low_limit, Rh_high_limit)
        ax.set_ylabel("Relative Humidity (%)", fontsize=18)
        ax.set_xlabel('Operative Temperature ($^\circ$C)', fontsize=18)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(air_low_limit, air_high_limit + offset)
        plt.xticks(np.arange(17, air_high_limit, 1.0))
        date_list = data["Date"].unique()
        marker_list = ["o", "*", "p", ">", "<", "d", "^", "x"]

        print(self.user)
        not_meet = data.loc[data[self.output] < 0]
        print(len(not_meet.index))
        print(len(data.index))
        print(len(not_meet.index)/len(data.index))

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
        lengend_list = [" Outdoor:(Ta$^\circ$C, Rh%)"]
        lengend_list = lengend_list + [str(date_list[i]) + ":" 
            + str(self.outdoor[i]) for i in range(len(date_list))]
        legend1 = plt.legend(custom_legend, lengend_list, loc='upper right',
         fontsize=12)  
        plt.gca().add_artist(legend1)

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
        ax.tick_params(labelsize = 18)
        plt.legend(custom_legend, lengend_list , loc='lower right',
         fontsize=12) 

        plt.tight_layout()
        plt.savefig(self.filename + "_" + plot_x + "_" + plot_y + "_" + date)
        plt.close()
  


    def combine_data(self):
        i = 0
        outdoor_list = []
        marker_list = ["o", "*", "p", ">", "<", "d", "^", "x"]
        for tag in self.tags:
            date = tag[6:]
            vote_File = self.location + "voting-" + tag + ".csv"
            skin_File = self.location + "temp-" + tag + ".csv"
            air_File = self.location + "environment-" + tag + ".csv"
            outdoor_File = self.location + "environment-outdoor-" + date + ".csv"
            heart_File = self.location + "hrate-" + tag + ".csv"
            RR_File = self.location + "rr-" + tag + ".csv"
            if i == 0:
                data, outdoor = self.process_data_Air_Skin_Sensation(skin_File, air_File, vote_File, outdoor_File, heart_File, RR_File)
                #self.analyze_data_Ta_Rh(data, date, marker_list[i])
            
            else:
                data_i, outdoor = self.process_data_Air_Skin_Sensation(skin_File, air_File, vote_File, outdoor_File, heart_File, RR_File)
                #self.analyze_data_Ta_Rh(data_i, date, marker_list[i])
                data = data.append(data_i)
            outdoor_list.append(outdoor)
            i += 1
        return data, outdoor_list



    def process_data_Air_Skin_Sensation(self, skin_File, air_File, vote_File, outdoor_File, heart_File, RR_File):
        ####### process outdoor

        data_outdoor = pd.read_csv(outdoor_File, names = ["Time", "Out_Temperature", "Out_Humidity"])
        outdoor =  (data_outdoor.mean().round()["Out_Temperature"], data_outdoor.mean().round()["Out_Humidity"])
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

        ##### process outdoor input 
        data_outdoor.index = pd.to_datetime(data_outdoor.Time)
        data_outdoor = data_outdoor.resample('60s').mean()
        data_outdoor_0 = pd.DataFrame(np.array([data_outdoor[(data_outdoor.index == t)].mean()
            for t in data_vote.index]), columns = ["Out_Temperature_0m", "Out_Humidity_0m"], index = data_vote.index)
        data_vote_skin_air_out = data_vote_skin_air.join(data_outdoor_0)

        ######process heart rate
        data_heart = pd.read_csv(heart_File, names = ["Time", "Heart"])
        data_heart.index = pd.to_datetime(data_heart.Time)
        data_heart = data_heart.resample('60s').mean()
        data_heart_0 = pd.DataFrame(np.array([data_heart[(data_heart.index == t)].mean()
            for t in data_vote.index]), columns = ["Heart_0m"], index = data_vote.index)
        data_vote_skin_air_out_heart = data_vote_skin_air_out.join(data_heart_0)
      

        ######process rr rate
        data_RR = pd.read_csv(RR_File, names = ["Time", "RR"])
        data_RR.index = pd.to_datetime(data_RR.Time)
        data_RR = data_RR.resample('60s').mean()
        data_RR_0 = pd.DataFrame(np.array([data_RR[(data_RR.index == t)].mean()
            for t in data_vote.index]), columns = ["RR_0m"], index = data_vote.index)
        data_vote_skin_air_out_heart_RR = data_vote_skin_air_out_heart.join(data_RR_0)
        return data_vote_skin_air_out_heart_RR, outdoor


def get_humidity_ratio(Ta, RH):
    # 1 atmosphere = 101325 pascal
    # 1 Celsius = 273.15 Kelvin 
    H_ratio = [humidity.rh2mixr(RH[i]/100.0, 101325, Ta[i] + 273.15) for i in range(len(Ta))]
    return H_ratio





def main():
    tags_hu = {"user4" :  ["user4-2018-2-19", "user4-2018-3-8", "user4-2018-3-10", "user4-2018-3-23"], 
    "user5" :  ["user5-2018-3-8", "user5-2018-3-10", "user5-2018-3-19","user5-2018-3-22", "user5-2018-3-25"],
     "user6": ["user6-2018-2-22", "user6-2018-2-24", "user6-2018-3-4", "user6-2018-3-21", "user6-2018-3-24"]}
    for i in range(5,5):
        user = "user" + str(i)
        rh_simulator = HumiditySimulator(user, tags_hu[user], False)
        rh_simulator.evaluation()

        rh_simulator = HumiditySimulator(user, tags_hu[user], True)
        rh_simulator.evaluation()

    inSampleTime = '2018-02-10 18:08:00'
    tags = {"user1" : ["user1-2018-2-20", "user1-2018-2-24", "user1-2018-3-2", "user1-2018-3-19", "user1-2018-3-24", "user1-2018-3-31", "user1-2018-4-6"],
    "user2": ["user2-2018-2-20", "user2-2018-2-25", "user2-2018-2-27", "user2-2018-3-2", "user2-2018-3-23", "user2-2018-3-25", "user2-2018-3-31", "user2-2018-4-6"], 
    "user3": ["user3-2018-2-19", "user3-2018-2-27", "user3-2018-3-2", "user3-2018-3-14", "user3-2018-3-16", "user3-2018-3-17", "user3-2018-3-24"],
    "user4": ["user4-2018-2-19", "user4-2018-2-25", "user4-2018-3-8", "user4-2018-3-10", "user4-2018-3-20", "user4-2018-3-23", "user4-2018-3-24"],
    "user5": ["user5-2018-3-4", "user5-2018-3-8", "user5-2018-3-10", "user5-2018-3-19", "user5-2018-3-20", "user5-2018-3-22", "user5-2018-3-25"], 
    "user6": ["user6-2018-2-22", "user6-2018-2-24", "user6-2018-3-4", "user6-2018-3-20", "user6-2018-3-21", "user6-2018-3-24", "user6-2018-4-11"]}

    fig, ax = plt.subplots(figsize=(8,6))
    color_list = ['gold', "darkred", 'olive', 'olive',   'coral', 'coral', "navy"] #'gold', 'olive', 'coral', 'C4', 'sienna', 'teal', 'crimson'#'#C0C0C0', '#A9A9A9', '#808080',  # #b', 'g', 'r', 'c', 'm', 'y', 'k'
    for i in [4]: #,2,3,4,5,6
        user = "user" + str(i)
        print(user)

        # rh_simulator = HumiditySimulator(user, tags_hu[user], False)
        # rh_simulator.evaluation()

        # rh_simulator = HumiditySimulator(user, tags_hu[user], True)
        # rh_simulator.evaluation()
        
        # skin_simulator = skinSimulator(user, tags[user], ["Air Temperature", "Relative Humidity"], "Skin Temperature", False)  
        # skin_simulator.evaluation()

        # skin_simulator = skinSimulator(user, tags[user], ["Air Temperature", "Relative Humidity"], "Skin Temperature", True)  
        # skin_simulator.evaluation()

        #train with real data
        sub_simulator = subjectiveSimulator(user, "Sensation", tags[user], False, False, True, (ax, color_list[i]) )
        #sub_simulator.svm()
        #sub_simulator.PMV()
        sub_simulator.randomForest()
        # #sub_simulator.neural_network("adam")
        sub_simulator.evaluation("Random_forest", False)

        # sub_simulator = subjectiveSimulator(user, "Sensation", tags[user], False, False, False)
        # sub_simulator.neural_network("adam")
        # sub_simulator.evaluation("neural_network")


        # sub_simulator = subjectiveSimulator(user, "Sensation", tags[user], False, False, True)
        # sub_simulator.neural_network("adam")
        # sub_simulator.evaluation("neural_network")

        # #train with PMV data
        # sub_simulator = subjectiveSimulator(user, "Sensation", tags[user], True, False, False)
        # sub_simulator.neural_network("adam")
        # sub_simulator.evaluation("neural_network")  

        # # train with real data using PMV model
        # sub_simulator = subjectiveSimulator(user, "Sensation", tags[user], False, True, False)
        # sub_simulator.neural_network("adam")
        # sub_simulator.evaluation("neural_network")

        # sub_simulator = subjectiveSimulator(user, "Sensation", tags[user], False, True, True)
        # sub_simulator.neural_network("adam")
        # sub_simulator.evaluation("neural_network")

        # sub_simulator = subjectiveSimulator(user, "Sensation", tags[user])
        # sub_simulator.neural_network("adam")
        # sub_simulator.evaluation("neural_network")
        
   # plt.savefig("csv/half16_satisfaction")
    # env_File = [location + "action_16.csv",
    #             location + "environment-outdoor-2018-3-6.csv",
    #             location + "environment-user2-2018-3-6.csv",
    #             location + "environment-user4-2018-3-6.csv",
    #             location + "environment-user6-2018-3-6.csv"]
    location = "csv/env/"
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


    # temp_simulator = TempSimulator(location, env_File, "user4 ", ["Previous Air Temperature", 
    #     "Previous Action" ,"outdoor Previous Air Temperature"], "Air Temperature", 
    #     location + " envSimulator")
    # env_simulator = envSimulator(location, env_File, "user4 ", ["Previous Relative Humidity", "Previous Action", 
    #     "outdoor Previous Relative Humidity"],
    #   "Relative Humidity", location + "envSimulator")
    # env_simulator.evaluation()
    
    #env_simulator.SARIMAX_prediction()

    # Our 2-dimensional distribution will be over variables X and Y
    # fig, ax = plt.subplots(figsize=(16,12))
    # N = 100
    # X = np.linspace(18, 30, N)
    # Y = np.linspace(10, 40, N)
    # X, Y = np.meshgrid(X, Y)

    # # Mean vector and covariance matrix
    # mu = np.array([22.8, 19.8])
    # Sigma = np.array([[ 1.29 , -0.5], [-0.5,  5.46]])

    # # Pack X and Y into a single 3-dimensional array
    # pos = np.empty(X.shape + (2,))
    # pos[:, :, 0] = X
    # pos[:, :, 1] = Y

    # def multivariate_gaussian(pos, mu, Sigma):
    #     """Return the multivariate Gaussian distribution on array pos.

    #     pos is an array constructed by packing the meshed arrays of variables
    #     x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    #     """

    #     n = mu.shape[0]
    #     Sigma_det = np.linalg.det(Sigma)
    #     Sigma_inv = np.linalg.inv(Sigma)
    #     N = np.sqrt((np.pi)**n * Sigma_det)
    #     # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    #     # way across all the input variables.
    #     fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    #     return np.exp(-fac / 2) / N

    # # The distribution on the variables X, Y packed into pos.
    # Z = multivariate_gaussian(pos, mu, Sigma)

    # # Create a surface plot and projected filled contour plot under it.
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
    #                 cmap=cm.viridis)

    # cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

    # # Adjust the limits, ticks and view angle
    # ax.set_zlim(-0.15,0.2)
    # ax.set_zticks(np.linspace(0,0.2,5))
    # ax.view_init(27, -21)
    # ax.set_zlabel("Probability")
    # ax.set_xlabel("Temperature (C)")
    # ax.set_ylabel("Humidity (%)")
    # plt.title("User3 Multivariate Gaussian")

    # plt.savefig("csv/shihao_satisfaction")


if __name__ == '__main__':
    main()

