from learning.baggingRegressor import baggingRegressor
from learning.gradientBoostingRegressor import gradientBoostingRegressor
from learning.sgdRegressor import sgdRegressor
from learning.randomForestRegressor import rfRegressor
from learning.stackingRegressor import stackingRegressor
from learning.linearRegressor import linearRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np

def performTraining(dataset):
    train_set, energy_consumption, test_set = generateTrainingData(dataset)
    models_list = createModels(train_set, energy_consumption)
    models_comparison = compareModels(train_set, energy_consumption, models_list)

    return models_comparison

def generateTrainingData(dataset):

    energy_consumption = dataset["mtedle2019_elec_conso_tot"]
    data = dataset.drop("mtedle2019_elec_conso_tot", axis=1)

    train_set, test_set, predict_train, predict_test = train_test_split(data, energy_consumption, test_size=0.2, random_state=42)

    return train_set, predict_train, test_set

def createModels(train_set, energy_consumption):
    models_list = []

    #print('\n Create bagging Model')
    #models_list.append(baggingRegressor(train_set, energy_consumption))

    #print('\n Create gradient boosting Model')
    #models_list.append(gradientBoostingRegressor(train_set, energy_consumption))
    
    print('\n Create sgd Model')
    models_list.append(sgdRegressor(train_set, energy_consumption))

    print('\n Create random forest Model')
    models_list.append(rfRegressor(train_set, energy_consumption))

    #print('\n Create stacking Model')
    #models_list.append(stackingRegressor(train_set, energy_consumption))

    print('\n Create linear Model')
    models_list.append(linearRegressor(train_set, energy_consumption))

    return models_list

def compareModels(train_set, energy_consumption, models_list):
    models_results = []

    for model in models_list:
        scores = cross_val_score(model, train_set, energy_consumption, scoring="neg_mean_squared_error", cv=10)
        
        model_scores = np.sqrt(-scores)
        results = [model_scores.mean(), model_scores.std()]

        models_results.append(results)

    return models_results

