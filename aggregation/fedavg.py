from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import time
import numpy as np

global_model = MLPRegressor(random_state=0, max_iter=10000)
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
    while i <= nb_loop:
        print("Federation round n°" + str(i))
        globals()['start_time'] = time.time()
        updateLocalModels()
        aggregated_weight_matrix, aggregated_bias_matrix = federate()
        updateGlobalModel(aggregated_weight_matrix, aggregated_bias_matrix)
        generateLoopMetrics(i)
        i = i+1
    printMetrics()

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
    #global_model.coefs_ = aggregated_weight_matrix
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
        average += np.array(elem, dtype=object)
        i+=1
    
    return np.true_divide(average, i).tolist()

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
    loop_duration = time.time() - start_time
    
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

    
