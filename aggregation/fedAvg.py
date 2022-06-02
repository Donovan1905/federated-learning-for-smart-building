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

global_model = MLPRegressor(hidden_layer_sizes=(15, ), activation='tanh', solver='adam', alpha=0.0001, random_state=42, max_iter=1000)
local_models = []
local_models_X_train_data = []
local_models_Y_train_data = []
local_models_X_test_data = []
local_models_Y_test_data = []

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

def generateTrainingData(floor, nb_loop):
    dataset_folder = os.path.join(os.path.dirname(__file__), "../_data/csv")
    dataset = pd.read_csv(dataset_folder + "/wrangled-floor-" + str(floor) +".csv")

    energy_consumption = dataset["tot_energy"]
    data = dataset.drop("tot_energy", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, energy_consumption, test_size=0.2, random_state=42)
    
    dfs_x = np.array_split(X_train, nb_loop)
    dfs_y = np.array_split(y_train, nb_loop)

    local_models_X_train_data.append(dfs_x)
    local_models_Y_train_data.append(dfs_y)

    local_models_X_test_data.append(X_test)
    local_models_Y_test_data.append(y_test)

    return X_train, X_test, y_train, y_test

def generateTestingData(testing_floor):
    dataset_folder = os.path.join(os.path.dirname(__file__), "../_data/csv")
    dataset = pd.read_csv(dataset_folder + "/wrangled-floor-" + str(testing_floor) +".csv")

    energy_consumption = dataset["tot_energy"]
    data = dataset.drop("tot_energy", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, energy_consumption, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def fedAvg(nb_loop, nb_clients, testing_floor):
    i = 1
    X_train_compare, X_test_compare, y_train_compare, y_test_compare = generateTestingData(testing_floor)
    generateLocalModels(nb_clients, nb_loop)
    while i <= nb_loop :
        print("Federation round n°" + str(i))
        globals()['start_time'] = time.time()
        aggregated_weight_matrix, aggregated_bias_matrix = federate()
        updateGlobalModel(aggregated_weight_matrix, aggregated_bias_matrix, X_train_compare, y_train_compare)
        updateLocalModels(i)
        generateLoopMetrics(i, X_test_compare, y_test_compare)
        i = i+1
    printMetrics()
    storeMetrics()

def generateLocalModels(nb_clients, nb_loop):
    i = 0
    while i <= nb_clients:
        X_train, X_test, y_train, y_test = generateTrainingData(i+3, nb_loop)

        model = MLPRegressor(hidden_layer_sizes=(15, ), activation='tanh', solver='adam', alpha=0.0001, random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        local_models.append(model)
        i = i+1

def updateGlobalModel(aggregated_weight_matrix, aggregated_bias_matrix, X_train_compare, y_train_compare):
    global_model.partial_fit(X_train_compare, y_train_compare)
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

    aggregated_bias_matrix = aggregate(bias_matrix)

    return aggregated_weight_matrix, aggregated_bias_matrix

def aggregate(matrix):
    average = matrix[0]
    i=1
    for elem in matrix[1:]:
        average += elem
        i+=1
    newList = [elem/i for elem in average]
    return newList


def updateLocalModels(loop_number):
    print("----- UPDATE LOCAL MODELS -----")
    i = 0
    for model in local_models:
        model_index = local_models.index(model)
        model = global_model
        model.fit(local_models_X_train_data[model_index][loop_number-1], local_models_Y_train_data[model_index][loop_number-1])
        print("Client n°" + str(i+1) + " updated")
        i = i+1

def generateLoopMetrics(loop_number, X_test_compare, y_test_compare):
    print("----- GENERATE METRICS -----")
    score = global_model.score(X_test_compare, y_test_compare)

    global_model_score_history.append(score)
    global_model_loss_history.append(global_model.best_loss_)
    loop_duration = (time.time() - globals()['start_time'])/60

    loop_max_err = []
    loop_mae = []
    loop_mse = []
    loop_rmse = []

    for model in local_models:
        model_id = local_models.index(model)
        y_pred = model.predict(local_models_X_test_data[model_id])
        loop_max_err.append(max_error(local_models_Y_test_data[model_id], y_pred))
        loop_mae.append(mean_absolute_error(local_models_Y_test_data[model_id], y_pred))
        loop_mse.append(mean_squared_error(local_models_Y_test_data[model_id], y_pred))
        loop_rmse.append(np.sqrt(mean_squared_error(local_models_Y_test_data[model_id], y_pred)))

    y_pred = global_model.predict(X_test_compare)
    max_err_history.append(max_error(y_test_compare, y_pred))
    mae_history.append(mean_absolute_error(y_test_compare, y_pred))
    mse_history.append(mean_squared_error(y_test_compare, y_pred))
    rmse_history.append(np.sqrt(mean_squared_error(y_test_compare, y_pred)))
    
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

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = os.path.join(os.path.dirname(__file__), "../_data/results/aggregation/fedAvg/fedAvg-" + str(date))
    os.mkdir(results_folder)
    
    results_df.to_csv(results_folder + '/global_results.csv')
    score_history_df.to_csv(results_folder + '/score_history.csv')
    loss_history_df.to_csv(results_folder + '/loss_history.csv') 
    max_err_history_df.to_csv(results_folder + '/max_err_history.csv') 
    mae_history_df.to_csv(results_folder + '/mae_history.csv') 
    mse_history_df.to_csv(results_folder + '/mse_history.csv') 
    rmse_history_df.to_csv(results_folder + '/rmse_history.csv') 
    time_history_df.to_csv(results_folder + '/time_history.csv') 