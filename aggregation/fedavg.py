from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
import time
from time import strftime
import numpy as np
import pandas as pd
import os
import datetime

global_model = MLPRegressor(random_state=0, max_iter=10000)
local_models = []
X, y = make_regression(n_samples=200, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
global_model_best_score = 0
global_model_best_loss = 0
global_model_best_round = 0 
fastest_loop_time = 0
fastest_loop = 0
best_global_model = None
start_time = 0

global_model_score_history = []
global_model_loss_history = []
loop_time_history = []
max_err_history = []
mae_history = []
mse_history = []
rmse_history = []


def federationLoop(nb_loop):
    i = 1
    generateLocalModels()
    while i <= nb_loop:
        print("Federation round n°" + str(i))
        globals()['start_time'] = time.time()
        updateLocalModels()
        aggregated_weight_matrix, aggregated_bias_matrix = federate()
        updateGlobalModel(aggregated_weight_matrix, aggregated_bias_matrix)
        generateLoopMetrics(i)
        i = i+1
    printMetrics()
    storeMetrics()

def generateLocalModels():
    i = 0
    while i < 5:
        model = MLPRegressor(random_state=0, max_iter=10000)
        model.fit(x_train, y_train)
        local_models.append(model)
        i = i+1

def updateGlobalModel(aggregated_weight_matrix, aggregated_bias_matrix):
    global_model.partial_fit(x_train, y_train)
    print("----- UPDATE GLOBAL MODEL -----")
    global_model.coefs_ = aggregated_weight_matrix
    global_model.intercepts_ = aggregated_bias_matrix
    print("Global model updated")

def federate():
    print("------ FEDERATE LOCAL MODELS ------")
    weight_matrix = []
    bias_matrix = []

    for model in local_models:
        weight_matrix.append(model.coefs_)
        bias_matrix.append(model.intercepts_)

    aggregated_weight_matrix = aggregate(weight_matrix)
    print("Aggregated weight matrix")
    print(aggregated_weight_matrix)

    aggregated_bias_matrix = aggregate(bias_matrix)
    print("Aggregated bias matrix")
    print(aggregated_bias_matrix)

    return aggregated_weight_matrix, aggregated_bias_matrix

def aggregate(matrix):
    average = matrix[0]
    i=1
    for elem in matrix[1:]:
        average += elem
        i+=1
    
    newList = [elem/i for elem in average]
    
    return newList


def updateLocalModels():
    print("----- UPDATE LOCAL MODELS -----")
    i = 1
    for model in local_models:
        model = global_model
        model.fit(x_train, y_train)
        print("Model n°" + str(i) + " updated")
        i = i+1

def generateLoopMetrics(loop_number):
    print("----- GENERATE METRICS -----")
    score = global_model.score(x_test, y_test)

    global_model_score_history.append(score)
    global_model_loss_history.append(global_model.best_loss_)
    loop_duration = (time.time() - globals()['start_time'])/60

    loop_max_err = []
    loop_mae = []
    loop_mse = []
    loop_rmse = []

    for model in local_models:
        y_pred = model.predict(x_test)
        loop_max_err.append(max_error(y_test, y_pred))
        loop_mae.append(mean_absolute_error(y_test, y_pred))
        loop_mse.append(mean_squared_error(y_test, y_pred))
        loop_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    max_err_history.append(min(loop_max_err))
    mae_history.append(min(loop_mae))
    mse_history.append(min(loop_mse))
    rmse_history.append(min(loop_rmse))
    
    if (loop_number==1):
        globals()['global_model_best_score'] = score
        globals()['global_model_best_loop'] = loop_number
        globals()['best_global_model'] = global_model
        globals()['fastest_loop'] = 1
        globals()['global_model_best_loss'] = global_model.best_loss_
        globals()['fastest_loop_time'] = loop_duration
    else:
        if(loop_duration < globals()['fastest_loop_time']):
            globals()['fastest_loop'] = loop_number
            globals()['fastest_loop_time'] = loop_duration
        if(global_model.best_loss_ < globals()['global_model_best_loss']):
            globals()['global_model_best_loss'] = global_model.best_loss_
        if (score > globals()['global_model_best_score']):
            globals()['global_model_best_score'] = score
            globals()['global_model_best_loop'] = loop_number
            globals()['best_global_model'] = global_model

    loop_time_history.append(loop_duration)

def printMetrics():
    print("\nBest score :")
    print(globals()['global_model_best_score'])
    print("\nBest loss :")
    print(globals()['global_model_best_loss'])
    print("\nFastest round duration :")
    print(globals()['fastest_loop_time'])
    print("\nModel best round id :")
    print(globals()['global_model_best_loop'])
    print("\nFastest round id :")
    print(globals()['fastest_loop'])

    print("\nBest model :")
    print(globals()['best_global_model'])

    print("\nScores history :")
    print(globals()['global_model_score_history'])
    print("\nLoss history :")
    print(globals()['global_model_loss_history'])
    print("\nDuration history :")
    print(globals()['loop_time_history'])
    print("\nMax error history :")
    print(globals()['max_err_history'])
    print("\nMAE history :")
    print(globals()['mae_history'])
    print("\nMSE history :")
    print(globals()['mse_history'])
    print("\nRMSE history :")
    print(globals()['rmse_history'])

def storeMetrics():
    print("----- STORE METRICS -----")
    results_df = pd.DataFrame({"Best score": [globals()['global_model_best_score']], "Best loss": [globals()['global_model_best_loss']], "Fastest round duration": [globals()['fastest_loop_time']], "Model best round id": [globals()['global_model_best_loop']], "Fastest round id": [globals()['fastest_loop']], "Best_model'": [globals()['best_global_model']]})
    score_history_df = pd.DataFrame(globals()['global_model_score_history'])
    loss_history_df = pd.DataFrame(globals()['global_model_loss_history'])
    max_err_history_df = pd.DataFrame(globals()['max_err_history'])
    mae_history_df = pd.DataFrame(globals()['mae_history'])
    mse_history_df = pd.DataFrame(globals()['mse_history'])
    rmse_history_df = pd.DataFrame(globals()['rmse_history'])
    time_history_df = pd.DataFrame(globals()['loop_time_history'])

    date = datetime.datetime.now()
    results_folder = os.path.join(os.path.dirname(__file__), "../_data/results/aggregation/fedAvg-" + date)
    
    results_df.to_csv(results_folder + '/global_results.csv')
    score_history_df.to_csv(results_folder + '/score_history.csv')
    loss_history_df.to_csv(results_folder + '/loss_history.csv') 
    max_err_history_df.to_csv(results_folder + '/max_err_history.csv') 
    mae_history_df.to_csv(results_folder + '/mae_history.csv') 
    mse_history_df.to_csv(results_folder + '/mse_history.csv') 
    rmse_history_df.to_csv(results_folder + '/rmse_history.csv') 
    time_history_df.to_csv(results_folder + '/time_history.csv') 
 