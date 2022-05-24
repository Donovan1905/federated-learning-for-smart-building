from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import time
import numpy as np

global_model = MLPRegressor(random_state=0, max_iter=1000)
local_models = []
X, y = make_regression(n_samples=200, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
global_model_best_score = 0
global_model_best_loss = 0
global_model_best_round = 0 
global_model_score_history = []
global_model_loss_history = []
loop_time_history = []
fastest_loop_time = 0
fastest_loop = 0
best_global_model = None
start_time = 0

def federationLoop(nb_loop):
    i = 1
    generateLocalModels()
    while i < nb_loop:
        start_time = time.time()
        updateLocalModels()
        aggregated_weight_matrix, aggregated_bias_matrix = federate()
        updateGlobalModel(aggregated_weight_matrix, aggregated_bias_matrix)
        generateLoopMetrics(i)
        i = i+1
 
def generateLocalModels():
    i = 0
    while i < 5:
        model = MLPRegressor(random_state=0, max_iter=1000)
        model.fit(x_train, y_train)
        local_models.append(model)
        i = i+1

def updateGlobalModel(aggregated_weight_matrix, aggregated_bias_matrix):
    global_model.partial_fit(x_train, y_train)
    print("----- UPDATE GLOBAL MODEL -----")
    #global_model.coefs_ = aggregated_weight_matrix
    global_model.intercepts_ = aggregated_bias_matrix

def federate():
    print("------ FEDERATE LOCAL MODELS ------")
    weight_matrix = local_models[0].coefs_
    bias_matrix = local_models[0].intercepts_

    for model in local_models[1:]:
        weight_matrix += model.coefs_
        bias_matrix += model.intercepts_

    aggregated_weight_matrix = np.true_divide([2, 4, 6], 2).tolist()
    print("------ AGGREGATED WEIGHT MATRIX ------")
    print(aggregated_weight_matrix)

    aggregated_bias_matrix = np.true_divide(np.array(bias_matrix), len(local_models)).tolist()
    print("------ AGGREGATED BIAS MATRIX ------")
    print(aggregated_bias_matrix)

    return aggregated_weight_matrix, aggregated_bias_matrix

def updateLocalModels():
    print("----- UPDATE LOCAL MODELS -----")
    i = 1
    for model in local_models:
        model = global_model
        model.fit(x_train, y_train)
        print("Model n°" + str(i) + " updated")
        i = i+1

def generateLoopMetrics(loop_number):
    score = global_model.score(x_test, y_test)

    if (score > global_model_best_score):
        global_model_best_score = score
        global_model_best_loop = loop_number
        best_global_model = global_model

    global_model_score_history.append(score)
    global_model_loss_history.append(global_model.best_loss_)
    loop_duration = time.time() - start_time

    if (loop_number=1):
        fastest_loop = 1
        global_model_best_loss = global_model.best_loss_
        fastest_loop_time = loop_duration
    else:
        if(loop_duration < fastest_loop_time):
            fastest_loop = loop_number
            fastest_loop_time = loop_duration
        if(global_model.best_loss_ < global_model_best_loss):
            global_model_best_loss = global_model.best_loss_

    loop_time_history.append(loop_duration)
    
