from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

global_model = MLPRegressor(random_state=0, max_iter=1000)
local_models = []
X, y = make_regression(n_samples=200, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
 
def generateLocalModels():
    i = 0
    while i < 5:
        model = MLPRegressor(random_state=0, max_iter=1000)
        model.fit(x_train, y_train)
        local_models.append(model)
        i = i+1

    federate()

def updateGlobalModel(aggregated_weight_matrix, aggregated_bias_matrix):
    global_model.partial_fit(x_train, y_train)
    print("----- UPDATE GLOBAL MODEL -----")
    #global_model.coefs_ = aggregated_weight_matrix
    global_model.intercepts_ = aggregated_bias_matrix
    
    print("Iterations : " + str(global_model.n_iter_))
    print("Layers : " + str(global_model.n_layers_))
    print("Best Loss : " + str(global_model.best_loss_))
    print("Score : " + str(global_model.score(x_test, y_test)))

    updateLocalModels()

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

    updateGlobalModel(aggregated_weight_matrix, aggregated_bias_matrix)

def updateLocalModels():
    print("----- UPDATE LOCAL MODELS -----")
    i = 1
    for model in local_models:
        model = global_model
        model.fit(x_train, y_train)
        print("Model n°" + str(i) + " updated")
        i = i+1

    

    
