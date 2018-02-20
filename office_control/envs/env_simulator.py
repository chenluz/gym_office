# the actions can be any kinds of control to building
from mpl_toolkits.mplot3d import Axes3D
from pandas import Series
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import neighbors
import statsmodels.api as sm
from datetime import datetime
import requests
from sklearn.metrics import mean_squared_error
import pickle


# ref: http://www.statsmodels.org/dev/examples/notebooks/generated/statespace_sarimax_stata.html

def plot_observation():
    # plot obervation
    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.plot(data_m.index, data_m['Temperature'], 'g')
    ax1.set_xlabel('Time (m)')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Temperature', color='g')
    ax1.tick_params('y', colors='g')

    ax2 = ax1.twinx()
    ax2.plot(data_m.index, data_m['Action'], 'r')
    ax2.set_ylabel('Action', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.savefig("obervation.png")
    return 


def SARIMAX_prediction(state, pre_state):
    # Fit the model
    #The order argument is a tuple of the form (AR specification, Integration order, MA specification)
    # Variables
    inSampleTime = '2018-02-10 18:08:00'
    initial = data_m.ix[: inSampleTime]
    initial_endog = initial[state]

    initial_exog = sm.add_constant(initial['Action'])

    # Fit the model
    mod = sm.tsa.statespace.SARIMAX(initial_endog, exog=initial_exog, order=(1,0,1))
    fit_res = mod.fit(disp=False)
    print(fit_res.summary())

    #http://nbviewer.jupyter.org/gist/ChadFulton/d744368336ef4bd02eadcea8606905b5
    update = data_m
    print(update)
    update_endog = update['Temperature']
    update_exog = sm.add_constant(update['Action'])
    update_mod = sm.tsa.statespace.SARIMAX(update_endog, exog=update_exog, order=(1,0,1))
    # update_mod.initialize_known(fit_res.predicted_state[:, -length], 
    #     fit_res.predicted_state_cov[:, :, -length])
    update_res = update_mod.filter(fit_res.params)
    print(update_res.summary())
    predict = update_res.get_prediction()
    predict_ci = predict.conf_int()
    startTime = '2018-02-10 14:18:00'

    true_Y =  data_m.ix[inSampleTime:, state]

    # Dynamic predictions
    predict_dy = update_res.get_prediction(dynamic=inSampleTime)
    predict_dy_ci = predict_dy.conf_int()

    MSE = mean_squared_error(true_Y, predict_dy.predicted_mean.ix[inSampleTime:])
    #Graph
    fig, ax = plt.subplots(figsize=(12,8))

    ax.set(title='SARIMAX (MES:%f)'% MSE, xlabel='Date', ylabel='Temperature')

    # Plot data points
    update.ix[startTime:, 'Temperature'].plot(ax=ax, style='o', label='Observed')

    # Plot predictions
    predict_dy.predicted_mean.ix[startTime:].plot(ax=ax, style='g', label='Dynamic forecast ' + inSampleTime)
    ci = predict_dy_ci.ix[startTime:]
    ax.fill_between(ci.index, ci.ix[:,0], ci.ix[:,1], color='g', alpha=0.1)

    predict.predicted_mean.ix[startTime:,].plot(ax=ax, style='r--', label='One-step-ahead forecast')
    ci = predict_ci.ix[startTime:,]
    ax.fill_between(ci.index, ci.ix[:,0], ci.ix[:,1], color='r', alpha=0.1)

    legend = ax.legend(loc='lower right')
    plt.savefig("SARIMAXRegression.png")
    return


def kNN_prediction(state, pre_state):
    inSampleTime = '2018-02-10 18:08:00'
    train_Y = data_m.ix[:inSampleTime,state].as_matrix()

    true_Y = data_m.ix[inSampleTime:, state].as_matrix()
    true_X = data_m.ix[inSampleTime:, ['Action', pre_state]].as_matrix()

    # train_X = np.reshape(data_m.ix[:inSampleTime, 'Pre_Temperature'].as_matrix(), (len(train_Y), 1))
    # test_X = np.reshape(data_m.ix[inSampleTime:,'Pre_Temperature'].as_matrix(), 
    #     (len(data_m.ix[inSampleTime:,'Pre_Temperature'].as_matrix()), 1))
    train_X = data_m.ix[:inSampleTime, ['Action', pre_state]].as_matrix()
    test_X = data_m[['Action', pre_state]].as_matrix()

    # #############################################################################
    # Fit regression model
    n_neighbors = 2

    knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
    y1 = knn.fit(train_X, train_Y).predict(test_X)
    pred_Y = knn.fit(train_X, train_Y).predict(true_X)
    MSE = mean_squared_error(true_Y, pred_Y)

    knn = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
    y2 = knn.fit(train_X, train_Y).predict(test_X)
    
    #Graph
    fig, ax = plt.subplots(figsize=(12,8))

    ax.set(title="KNeighborsRegressor (k = %i) (MSE:%f)" % (n_neighbors ,MSE), 
        xlabel='Date', ylabel='Temperature')

    # Plot data points
    data_m['Temperature'].plot(ax=ax, style='o', label='Observed')

    # Plot predictions
    pd.DataFrame({"Uniform": y1.flatten()}, index = data_m.index).plot(ax=ax, style='r')  
    pd.DataFrame({'Distance': y2.flatten()}, index = data_m.index).plot(ax=ax, style='g')
    ax.axvline(inSampleTime, color='k', linestyle='--')

    # ax2 = ax.twinx()
    # data_m.ix[inSampleTime:,'Action'].plot(ax=ax2, style='r', label='Action')

    legend = ax.legend(loc='lower right')      
                                                        
    plt.savefig("KNNRegresiion.png")



def transition_probability():
     # Set up the X and Y dimensions
    data_m['D_Temperature'] = pd.cut(data_m['Temperature'],  np.arange(22, 27, 0.1))
    data_m['D_Pre_Temperature'] = pd.cut(data_m['Pre_Temperature'],  np.arange(22, 27, 0.1))

    # get number of state at that action: 
    action_Dict = {}
    for action in data_m['Action'].unique(): 
        state_Dict = {}
        for state in data_m['D_Temperature'].unique():
            state_Dict[state] = (data_m['D_Temperature'] == state).sum()
        action_Dict[action] = state_Dict

    transit_Dict = {}
    for action in data_m['Action'].unique(): 
        pre_state_Dict = {}
        for pre_state in data_m['D_Pre_Temperature'].unique():
            state_Dict = {}
            for state in data_m['D_Temperature'].unique():
                state_Dict[state] = (((data_m['D_Temperature'] == state)&(data_m['D_Pre_Temperature'] 
                    == pre_state)).sum())*1.0/action_Dict[action][state]
            pre_state_Dict[pre_state] = state_Dict
        transit_Dict[action] = pre_state_Dict      
    return transit_Dict



def get_highest(pre_state, action):
    data_m['D_Pre_Temperature'] = pd.cut(data_m['Pre_Temperature'],  np.arange(22, 27, 0.1))
    for pre_state_interval in data_m['D_Pre_Temperature'].unique():
        if pre_state in pre_state_interval:
            pre = pre_state_interval
            break
    transit_Dict = transition_probability()
    for key in transit_Dict[action][pre]:
        print(key , transit_Dict[action][pre][key])



def MMDP():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    inSampleTime = '2018-02-10 18:00:00'
    Y = data_m.ix[:inSampleTime,'Temperature'].as_matrix()

    X = np.reshape(data_m.ix[:inSampleTime, 'Pre_Temperature'].as_matrix(), (len(Y), 1))
    train_X = data_m.ix[:inSampleTime, ['Action', 'Pre_Temperature']].as_matrix()
    X, Y = np.meshgrid(X, Y)

    # Create the univarate normal coefficients
    # of intercep and slope, as well as the
    # conditional probability density
    Z = norm.pdf(Y, X, 1.0)

    # Plot the surface with the "coolwarm" colormap
    surf = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False
    )

    # Set the limits of the z axis and major line locators
    ax.set_zlim(0, 0.4)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Label all of the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P(Y|X)')

    # Adjust the viewing angle and axes direction
    ax.view_init(elev=30., azim=50.0)
    ax.invert_xaxis()
    ax.invert_yaxis()

    # Plot the probability density
    plt.show()



def KernelRidgeRegression(state, pre_state):
    inSampleTime = '2018-02-10 18:08:00'
    print(data_m)
    train_Y = data_m.ix[:inSampleTime, state].as_matrix()
    true_Y = data_m.ix[inSampleTime:, state].as_matrix()
    true_X = data_m.ix[inSampleTime:, ['Action', pre_state]].as_matrix()

    # train_X = np.reshape(data_m.ix[:inSampleTime, 'Pre_Temperature'].as_matrix(), (len(train_Y), 1))
    # test_X = np.reshape(data_m['Pre_Temperature'].as_matrix(), 
    #      (len(data_m['Pre_Temperature'].as_matrix()), 1))
    train_X = data_m.ix[:inSampleTime, ['Action', pre_state]].as_matrix()
    test_X = data_m[['Action', pre_state]].as_matrix()

    # #############################################################################
    # Fit regression model
    clf = KernelRidge(alpha=2.0)
    model = clf.fit(train_X, train_Y)
    # save the model to disk
    filename = 'kernel_regression_model_humidity.sav'
    pickle.dump(model, open(filename, 'wb'))

    y1 = model.predict(test_X)
    pred_Y = model.predict(true_X)

    MSE = mean_squared_error(true_Y, pred_Y)

    #Graph
    fig, ax = plt.subplots(figsize=(12,8))

    ax.set(title="KernelRidge Raw Data (MSE:%f)" % MSE, xlabel='Time', ylabel=state)

    # Plot data points
    data_m[state].plot(ax=ax, style='o', label='Observed')

    # Plot predictions
    pd.DataFrame({"KernelRidge": y1.flatten()}, index = data_m.index).plot(ax=ax, style='r')  
    ax.axvline(inSampleTime, color='k', linestyle='--')

    # ax2 = ax.twinx()
    # data_m['Action'].plot(ax=ax2, style='g', label='Action')

    legend = ax.legend(loc='lower right')      
                                                        
    plt.savefig("kernelRidge_humidity.png")

def process_data_Skin_Air(skinFile, airFile):
    data_skin = pd.read_csv('environment.csv')
    data_air = 

def main():
    data = pd.read_csv('environment.csv')
    # Dataset
    data.index = pd.to_datetime(data.Time)
    data_m = data#.resample('2T').mean()

if __name__ == '__main__':
    main()
