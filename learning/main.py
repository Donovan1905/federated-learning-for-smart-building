from learning.gradientBoostingRegressor import gradientBoostingRegressor
from learning.mlpRegressor import mlpRegressor
from learning.sgdRegressor import sgdRegressor
from learning.randomForestRegressor import rfRegressor
from learning.kneighborsRegressor import kneighborsRegressor
from learning.decisionTreeRegressor import decisionTreeRegressor
from learning.linearRegressor import linearRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
import numpy as np
import os
import pandas as pd
import datetime
from joblib import load

def performTraining(floor, testing_floor):
    X_train, X_test, y_train, y_test = generateTrainingData(floor)
    X_train_compare, X_test_compare, y_train_compare, y_test_compare = generateTestingData(testing_floor)
    #models_list = loadModels()
    models_list = createModels(X_train, y_train)
    compareModels(X_test_compare, y_test_compare, models_list, floor)

def generateTrainingData(floor):
    dataset_folder = os.path.join(os.path.dirname(__file__), "../_data/csv")
    dataset = pd.read_csv(dataset_folder + "/wrangled-floor-" + str(floor) +".csv")

    energy_consumption = dataset["tot_energy"]
    data = dataset.drop("tot_energy", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, energy_consumption, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def generateTestingData(testing_floor):
    dataset_folder = os.path.join(os.path.dirname(__file__), "../_data/csv")
    dataset = pd.read_csv(dataset_folder + "/wrangled-floor-" + str(testing_floor) +".csv")

    energy_consumption = dataset["tot_energy"]
    data = dataset.drop("tot_energy", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, energy_consumption, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def createModels(X_train, y_train):
    print("----- CREATE MODELS -----")
    models_list = []
    
    models_list.append(('GradientBoost', gradientBoostingRegressor(X_train, y_train)))
    models_list.append(('SGD', sgdRegressor(X_train, y_train)))
    models_list.append(('RandomForest', rfRegressor(X_train, y_train)))
    models_list.append(('KNeighbors', kneighborsRegressor(X_train, y_train)))
    models_list.append(('DecisionTree', decisionTreeRegressor(X_train, y_train)))
    models_list.append(('Linear', linearRegressor(X_train, y_train)))
    models_list.append(('Neural Network', mlpRegressor(X_train, y_train)))

    return models_list

def compareModels(X_test, y_test, models_list, floor):
    print("----- RUNNING MODEL COMPARISON -----")
    models_results = []
    names = []
    results_df = pd.DataFrame(columns=['Model', 'Score_mean', 'Score_std', 'Max_error', 'MAE', 'MSE', 'RMSE'])
    print(models_list)

    for name, model in models_list:
        cv_results = cross_val_score(model, X_test, y_test, scoring="neg_mean_squared_error", cv=10)
        model_scores = np.sqrt(-cv_results)
        
        y_pred = model.predict(X_test)
        max_err = max_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        models_results.append(cv_results)
        names.append(name)
        results = "\n %s: \nmean : %f \nstd : %f \nmax error : %f \nmae  : %f \nmse : %f \nrmse : %f" % (name, model_scores.mean(), model_scores.std(), max_err, mae, mse, rmse)

        array_of_results = [name, model_scores.mean(), model_scores.std(), max_err, mae, mse, rmse]
        dfInsert(results_df, array_of_results)

        print(results)

    print("----- STORE MODEL COMPARISON -----")
    results_folder = os.path.join(os.path.dirname(__file__), "../_data/results/learning")
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_df.to_csv(results_folder + '/floor-' + str(floor) + '-' + str(date) + '.csv') 

def loadModels():
    print("----- LOADING MODELS -----")
    models_folder = os.path.join(os.path.dirname(__file__), "../_data/models")
    models_list = []
    for model in os.listdir(models_folder):
        models_list.append(load(models_folder + '/' + model)) 
        print(str(model) + " loaded")
    return models_list

def dfInsert(df, row):
    insert_loc = df.index.max()

    if pd.isna(insert_loc):
        df.loc[0] = row
    else:
        df.loc[insert_loc + 1] = row