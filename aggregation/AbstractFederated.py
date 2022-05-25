from abc import ABCMeta, abstractmethod

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import time
import numpy as np


class AbstractFederated(metaclass=ABCMeta):
    global_model = MLPRegressor(random_state=0, max_iter=10000)
    local_models = []
    X, y = make_regression(n_samples=200, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
    global_model_best_score = 0
    global_model_best_loss = 0
    global_model_best_round = 0
    global_model_best_loop = 0
    global_model_score_history = []
    global_model_loss_history = []
    loop_time_history = []
    fastest_loop_time = 0
    fastest_loop = 0
    best_global_model = None
    start_time = 0

    @abstractmethod
    def federationLoop(self, nb_loop):
        i = 1
        self.generateLocalModels()
        while i <= nb_loop:
            print("Federation round n°" + str(i))
            self.start_time = time.time()
            self.updateLocalModels()
            aggregated_weight_matrix, aggregated_bias_matrix = self.federate()
            self.updateGlobalModel(aggregated_weight_matrix, aggregated_bias_matrix)
            self.generateLoopMetrics(i)
            i = i + 1
        self.printMetrics()

    @abstractmethod
    def federate(self):
        pass

    @abstractmethod
    def updateGlobalModel(self, aggregated_weight_matrix, aggregated_bias_matrix):
        self.global_model.partial_fit(self.x_train, self.y_train)
        print("----- UPDATE GLOBAL MODEL -----")
        # global_model.coefs_ = aggregated_weight_matrix
        self.global_model.intercepts_ = aggregated_bias_matrix
        print("Global model updated")

    @abstractmethod
    def updateLocalModels(self):
        print("----- UPDATE LOCAL MODELS -----")
        i = 1
        for model in self.local_models:
            model = self.global_model
            model.fit(self.x_train, self.y_train)
            print("Model n°" + str(i) + " updated")
            i = i + 1

    @abstractmethod
    def generateLoopMetrics(self, loop_number):
        print("----- GENERATE METRICS -----")
        score = self.global_model.score(self.x_test, self.y_test)

        self.global_model_score_history.append(score)
        self.global_model_loss_history.append(self.global_model.best_loss_)
        loop_duration = time.time() - self.start_time

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

    @abstractmethod
    def generateLocalModels(self):
        i = 0
        while i < 5:
            model = MLPRegressor(random_state=0, max_iter=10000)
            model.fit(self.x_train, self.y_train)
            self.local_models.append(model)
            i = i + 1

    @abstractmethod
    def aggregate(self, matrix):
        pass
