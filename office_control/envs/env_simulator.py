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
    print(data[param2])
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



def KernelRidgeRegression(data_m, input_list, output, filename, inSampleTime):
    """
    Parameters:
    data_m :dataframe, the dataframe that saved all the all the X and Y 
    input: array, a array of column name in the DataFrame that used as input
    output: str, a column name in the DataFrame that used as output
    filename: str, specify the model name and png name that will be saved 
    inSampleTime: str, the time to split input and output
    """
 
    train_Y = data_m.ix[:inSampleTime, output].as_matrix()
    true_Y = data_m.ix[inSampleTime:, output].as_matrix()
    true_X = data_m.ix[inSampleTime:, input_list].as_matrix()

    # train_X = np.reshape(data_m.ix[:inSampleTime, 'Pre_Temperature'].as_matrix(), (len(train_Y), 1))
    # test_X = np.reshape(data_m['Pre_Temperature'].as_matrix(), 
    #      (len(data_m['Pre_Temperature'].as_matrix()), 1))
    train_X = data_m.ix[:inSampleTime, input_list].as_matrix()
    test_X = data_m[input_list].as_matrix()

    # #############################################################################
    # Fit regression model
    clf = KernelRidge(alpha=2.0)
    model = clf.fit(train_X, train_Y)
    # save the model to disk
    modlename = filename +'.sav'
    pickle.dump(model, open(modlename, 'wb'))

    y1 = model.predict(test_X)
    pred_Y = model.predict(true_X)

    MSE = mean_squared_error(true_Y, pred_Y)

    #Graph
    fig, ax = plt.subplots(figsize=(12,8))

    ax.set(title="KernelRidge Raw Data (MSE:%f)" % MSE, xlabel='Time', ylabel=output)

    # Plot data points
    data_m[output].plot(ax=ax, style='o', label='Observed')

    # Plot predictions
    pd.DataFrame({"KernelRidge": y1.flatten()}, index = data_m.index).plot(ax=ax, style='r')  
    ax.axvline(inSampleTime, color='k', linestyle='--')

    # ax2 = ax.twinx()
    # data_m['Action'].plot(ax=ax2, style='g', label='Action')

    legend = ax.legend(loc='lower right')      
                                                        
    plt.savefig(filename + ".png")

def KernelRidgeRegression_skin(data_m, input_list, output, filename, tag, is_test):
    """
    Parameters:
    data_m :dataframe, the dataframe that saved all the all the X and Y 
    input: array, a array of column name in the DataFrame that used as input
    output: str, a column name in the DataFrame that used as output
    filename: str, specify the model name and png name that will be saved 
    is_test: boolean, whether it is training or testing
    """
 
    true_Y = data_m[output].as_matrix()
    true_X = data_m[input_list].as_matrix()

    # #############################################################################
    if is_test == False: 
        # training and saving the modle
        # Fit regression model
        clf = KernelRidge(alpha=2.0)
        model = clf.fit(true_X, true_Y)
        # save the model to disk
        modlename = filename +'.sav'
        pickle.dump(model, open(modlename, 'wb'))
    else: 
        # load the trained model 
        model = pickle.load(open("csv/user3/Skin_Environment_user3-2018-2-19.sav", "rb"))

    pred_Y = model.predict(true_X)

    MSE = mean_squared_error(true_Y, pred_Y)

    #Graph
    fig, ax = plt.subplots(figsize=(12,8))

    ax.set(title="KernelRidge Raw Data (MSE:%f)" % MSE, xlabel='Time', ylabel=output)

    # Plot data points
    data_m[output].plot(ax=ax, style='o', label='Observed')

    # Plot predictions
    pd.DataFrame({"KernelRidge": pred_Y.flatten()}, index = data_m.index).plot(ax=ax, style='r')  

    legend = ax.legend(loc='lower right')      
                                                        
    plt.savefig(filename + ".png")
    plt.close()
    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_xlim(19, 31)
    ax.set_xlabel('Air Temperature(C)')
    ax.set_ylim(28, 37)
    ax.set_ylabel('Skin Temperature(C)')
    ax.plot(true_X, true_Y, "g.", label='Observed')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(true_X, pred_Y.flatten(), "r.", label='Predicted') 
    plt.title(tag)
    legend = ax.legend(loc='lower right')     
    plt.savefig(filename + "_relation")
    return


def neural_network(data_m, filename, is_test):
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
    print(dummy_y)

    model = Sequential()
    model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    model.fit(true_X_scaled, dummy_y, epochs=200, batch_size=5, verbose=0)
    classes = model.predict(true_X_scaled, batch_size=5)
    print(classes)



def process_data_Sen_Sat(vote_file):
    data_vote = pd.read_csv(vote_file)
    data_vote.index = pd.to_datetime(data_vote.Time)
    data_vote = data_vote.resample('300s').mean()
    return data_vote.dropna()


def process_data_Skin_Air(skin_File, Envir_File):
    """"
    Prcoess Air temperature and Relative Humidity 
    Temperature and Humidity is 10s interval, 
    Make every Temperature and Humidity within every 30s as 3 different features 

    Merge Skin and Environment Data
    """

    ####### process skin temperature
    data_skin = pd.read_csv(skin_File)
    data_skin.index = pd.to_datetime(data_skin.Time)
    data_skin = data_skin.resample('30s', closed="right").mean()
    print(data_skin)

    # ####### process air temperature and relative humidity
    data_air = pd.read_csv(Envir_File)
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
    return training_set.dropna()#.ix[:"2018-02-25 16:25:00"]

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
    # index1 = training_set.index.get_loc("2018-02-19 21:00:30" )
    # index2 = training_set.index.get_loc("2018-02-19 21:28:00" )
    # training_set = training_set.drop(training_set.index[range(index1,index2)])
    # with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    #     print(training_set)
    return training_set

def process_data_Air_Skin_Sensation(skin_File, Envir_File, vote_file):
    ####### process voting
    data_vote = pd.read_csv(vote_file)
    data_vote.index = pd.to_datetime(data_vote.Time)
    #print(data_vote)
    ####### process skin temperature
    data_skin = pd.read_csv(skin_File)
    data_skin.index = pd.to_datetime(data_skin.Time)
    data_skin = data_skin.resample('60s', closed="right").mean()
    data_skin_0 = pd.DataFrame(np.array([data_skin[(data_skin.index == t)].mean() 
        for t in data_vote.index]), columns = ["Skin_0m"], index = data_vote.index)
    data_skin_1 = pd.DataFrame(np.array([data_skin[(data_skin.index == t - dt.timedelta(minutes = 1))].mean() 
        for t in data_vote.index]), columns = ["Skin_1m"], index = data_vote.index)
    data_skin_2 = pd.DataFrame(np.array([data_skin[(data_skin.index == t - dt.timedelta(minutes = 2))].mean() 
        for t in data_vote.index]), columns = ["Skin_2m"], index = data_vote.index)
    data_vote_skin = data_vote.join(data_skin_0)
    data_vote_skin = data_vote_skin.join(data_skin_1)
    data_vote_skin = data_vote_skin.join(data_skin_2)

    ####### process air temperature and relative humidity
    data_air = pd.read_csv(Envir_File)
    data_air.index = pd.to_datetime(data_air.Time)
    # empty item filled with the value after it
    data_air= data_air.resample('60s').mean().bfill()
    data_air_0 = pd.DataFrame(np.array([data_air[(data_air.index == t)].mean()
        for t in data_vote.index]), columns = ["Temperature_0m", "Humidity_0m"], index = data_vote.index)
    data_air_1 = pd.DataFrame(np.array([data_air[(data_air.index == t - dt.timedelta(minutes = 1))].mean()  
        for t in data_vote.index]), columns = ["Temperature_1m", "Humidity_1m"], index = data_vote.index)
    data_air_2 = pd.DataFrame(np.array([data_air[(data_air.index == t - dt.timedelta(minutes = 2))].mean()  
        for t in data_vote.index]), columns = ["Temperature_2m", "Humidity_2m"], index = data_vote.index)
    data_vote_skin_air = data_vote_skin.join(data_air_0)
    data_vote_skin_air = data_vote_skin_air.join(data_air_1)
    data_vote_skin_air = data_vote_skin_air.join(data_air_2)
    return data_vote_skin_air.dropna()

def process_data_Action_Air():
    data = pd.read_csv('environment.csv')
    # Dataset
    data.index = pd.to_datetime(data.Time)
    data_m = data.resample('2T').mean()


def main():
    location = "csv/user3/"
    tag = "user3-2018-2-19"
    inSampleTime = "2018-02-24 22:00:00"
    vote_File = location + "voting-" + tag + ".csv"
    skin_File = location + "temp-" + tag + ".csv"
    envir_File = location + "environment-" + tag + ".csv"
    #data_m = process_data_Sen_Sat(vote_file)
    #plot_sat_sen(data_m, location + "Voting_" + tag, "Satisfaction", "Sensation", tag)
    #data_m = process_data_Skin_Air(skin_File, envir_File)
    #plot_relationship(data_m, location + "Voting_" + tag, "Sensation", "Satisfaction")
    #plot_observation(data_m, location + "Skin_Temperature_" + tag, "Temperature", "Skin", False, tag, 29, 37)
    #data_m = process_data_Air_Sensation(envir_File,vote_File)
    #print(data_m)
    #plot_observation(data_m, location + "Temperature_Humidity_" + tag, "Temperature", "Humidity", False, tag)
    #plot_observation(data_m, location + "Temperature_Sensation_" + tag, "Temperature", "Sensation", True, tag, -3.5, 3.5)
    #plot_observation(data_m, location + "Temperature_Satisfaction_" + tag, "Temperature", "Satisfaction", False, tag)
   
    # KernelRidgeRegression_skin(data_m, [
    #  ("Temperature")], "Skin", location + "Skin_Environment_" + tag, tag, True)
    data_m = process_data_Air_Skin_Sensation(skin_File, envir_File, vote_File)
    neural_network(data_m, location + "Sensation_" + tag, False)
    

if __name__ == '__main__':
    main()

