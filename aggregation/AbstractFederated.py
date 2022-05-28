from abc import ABCMeta, abstractmethod

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


class AbstractFederated(metaclass=ABCMeta):
    global_model = MLPRegressor(random_state=0, max_iter=10000)
    local_models = []
    local_models_X_data = []
    local_models_Y_data = []

    global_model_best_score = 0
    global_model_best_loss = 0
    global_model_best_round = 0
    global_model_best_loop = 0
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

    @abstractmethod
    def generateTrainingData(self, floor):
        dataset_folder = os.path.join(os.path.dirname(__file__), "../_data/csv")
        dataset = pd.read_csv(dataset_folder + "/wrangled-floor-" + str(floor) + ".csv")

        energy_consumption = dataset["tot_energy"]
        data = dataset.drop("tot_energy", axis=1)

        X_train, X_test, y_train, y_test = train_test_split(data, energy_consumption, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    @abstractmethod
    def generateTestingData(self, testing_floor):
        dataset_folder = os.path.join(os.path.dirname(__file__), "../_data/csv")
        dataset = pd.read_csv(dataset_folder + "/wrangled-floor-" + str(testing_floor) + ".csv")

        energy_consumption = dataset["tot_energy"]
        data = dataset.drop("tot_energy", axis=1)

        X_train, X_test, y_train, y_test = train_test_split(data, energy_consumption, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    @abstractmethod
    def federationLoop(self, nb_loop, nb_clients, testing_floor):
        i = 1
        X_train_compare, X_test_compare, y_train_compare, y_test_compare = self.generateTestingData(testing_floor)
        self.generateLocalModels(nb_clients)
        while i <= nb_loop:
            print("Federation round n°" + str(i))
            self.start_time = time.time()
            aggregated_weight_matrix, aggregated_bias_matrix = self.federate()
            self.updateGlobalModel(aggregated_weight_matrix, aggregated_bias_matrix, X_train_compare, y_train_compare)
            self.updateLocalModels()
            self.generateLoopMetrics(i, X_test_compare, y_test_compare)
            i = i + 1
        self.printMetrics()
        self.storeMetrics()

    @abstractmethod
    def federate(self):
        pass

    @abstractmethod
    def updateGlobalModel(self, aggregated_weight_matrix, aggregated_bias_matrix, X_train_compare, y_train_compare):
        self.global_model.partial_fit(X_train_compare, y_train_compare)
        print("----- UPDATE GLOBAL MODEL -----")
        self.global_model.coefs_ = aggregated_weight_matrix
        self.global_model.intercepts_ = aggregated_bias_matrix
        print("Global model updated")

    @abstractmethod
    def updateLocalModels(self):
        print("----- UPDATE LOCAL MODELS -----")
        i = 0
        for model in self.local_models:
            model_index = self.local_models.index(model)
            model = self.global_model
            model.fit(self.local_models_X_data[model_index], self.local_models_Y_data[model_index])
            print("Client n°" + str(i + 1) + " updated")
            i = i + 1

    @abstractmethod
    def generateLoopMetrics(self, loop_number, X_test_compare, y_test_compare):
        print("----- GENERATE METRICS -----")
        score = self.global_model.score(X_test_compare, y_test_compare)

        self.global_model_score_history.append(score)
        self.global_model_loss_history.append(self.global_model.best_loss_)
        loop_duration = (time.time() - self.start_time) / 60

        loop_max_err = []
        loop_mae = []
        loop_mse = []
        loop_rmse = []

        for model in self.local_models:
            y_pred = model.predict(X_test_compare)
            loop_max_err.append(max_error(y_test_compare, y_pred))
            loop_mae.append(mean_absolute_error(y_test_compare, y_pred))
            loop_mse.append(mean_squared_error(y_test_compare, y_pred))
            loop_rmse.append(np.sqrt(mean_squared_error(y_test_compare, y_pred)))

        self.max_err_history.append(min(loop_max_err))
        self.mae_history.append(min(loop_mae))
        self.mse_history.append(min(loop_mse))
        self.rmse_history.append(min(loop_rmse))

        if (loop_number == 1):
            self.global_model_best_score = score
            self.global_model_best_loop = loop_number
            self.best_global_model = self.global_model
            self.fastest_loop = 1
            self.global_model_best_loss = self.global_model.best_loss_
            self.fastest_loop_time = loop_duration
        else:
            if (loop_duration < self.fastest_loop_time):
                self.fastest_loop = loop_number
                self.fastest_loop_time = loop_duration
            if (self.global_model.best_loss_ < self.global_model_best_loss):
                self.global_model_best_loss = self.global_model.best_loss_
            if (score > self.global_model_best_score):
                self.global_model_best_score = score
                self.global_model_best_loop = loop_number
                self.best_global_model = self.global_model

        self.loop_time_history.append(loop_duration)

    @abstractmethod
    def printMetrics(self):
        print("\nBest score :")
        print(self.global_model_best_score)
        print("\nBest loss :")
        print(self.global_model_best_loss)
        print("\nFastest round duration :")
        print(self.fastest_loop_time)
        print("\nModel best round id :")
        print(self.global_model_best_loop)
        print("\nFastest round id :")
        print(self.fastest_loop)

        print("\nBest model :")
        print(self.best_global_model)

        print("\nScores history :")
        print(self.global_model_score_history)
        print("\nLoss history :")
        print(self.global_model_loss_history)
        print("\nDuration history :")
        print(self.loop_time_history)
        print("\nMax error history :")
        print(self.max_err_history)
        print("\nMAE history :")
        print(self.mae_history)
        print("\nMSE history :")
        print(self.mse_history)
        print("\nRMSE history :")
        print(self.rmse_history)

    @abstractmethod
    def generateLocalModels(self, nb_clients):
        i = 0
        while i <= nb_clients:
            X_train, X_test, y_train, y_test = self.generateTrainingData(i + 3)
            self.local_models_X_data.append(X_train)
            self.local_models_Y_data.append(y_train)
            model = MLPRegressor(random_state=0, max_iter=10000)
            model.fit(X_train, y_train)
            self.local_models.append(model)
            i = i + 1

    @abstractmethod
    def aggregate(self, matrix):
        pass

    @abstractmethod
    def storeMetrics(self):
        print("----- STORE METRICS -----")
        results_df = pd.DataFrame(
            {"Best score": [self.global_model_best_score], "Best loss": [self.global_model_best_loss],
             "Fastest round duration": [self.fastest_loop_time],
             "Model best round id": [self.global_model_best_loop], "Fastest round id": [self.fastest_loop],
             "Best_model'": [self.best_global_model]})
        score_history_df = pd.DataFrame(self.global_model_score_history)
        loss_history_df = pd.DataFrame(self.global_model_loss_history)
        max_err_history_df = pd.DataFrame(self.max_err_history)
        mae_history_df = pd.DataFrame(self.mae_history)
        mse_history_df = pd.DataFrame(self.mse_history)
        rmse_history_df = pd.DataFrame(self.rmse_history)
        time_history_df = pd.DataFrame(self.loop_time_history)

        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_folder = os.path.join(os.path.dirname(__file__), "../_data/results/aggregation/fedAvg-" + str(date))
        os.mkdir(results_folder)

        results_df.to_csv(results_folder + '/global_results.csv')
        score_history_df.to_csv(results_folder + '/score_history.csv')
        loss_history_df.to_csv(results_folder + '/loss_history.csv')
        max_err_history_df.to_csv(results_folder + '/max_err_history.csv')
        mae_history_df.to_csv(results_folder + '/mae_history.csv')
        mse_history_df.to_csv(results_folder + '/mse_history.csv')
        rmse_history_df.to_csv(results_folder + '/rmse_history.csv')
        time_history_df.to_csv(results_folder + '/time_history.csv')