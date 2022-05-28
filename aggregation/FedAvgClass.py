from abc import ABC

import numpy as np

from aggregation.AbstractFederated import AbstractFederated


class FedAvgClass(AbstractFederated, ABC):
    def __init__(self, nb_loop, nb_clients, testing_floor):
        print("FedAvg created")
        self.federationLoop(nb_loop, nb_clients, testing_floor)

    def generateTrainingData(self, floor):
        return super().generateTrainingData(floor)

    def generateTestingData(self, testing_floor):
        return super().generateTestingData(testing_floor)

    def federationLoop(self, nb_loop, nb_clients, testing_floor):
        super().federationLoop(nb_loop, nb_clients, testing_floor)

    def federate(self):
        print("------ FEDERATE LOCAL MODELS ------")
        weight_matrix = []
        bias_matrix = []

        for model in self.local_models:
            weight_matrix.append(model.coefs_)
            bias_matrix.append(model.intercepts_)

        aggregated_weight_matrix = self.aggregate(weight_matrix)
        print("Aggregated weight matrix")
        print(aggregated_weight_matrix)

        aggregated_bias_matrix = self.aggregate(bias_matrix)
        print("Aggregated bias matrix")
        print(aggregated_bias_matrix)

        return aggregated_weight_matrix, aggregated_bias_matrix

    def updateGlobalModel(self, aggregated_weight_matrix, aggregated_bias_matrix, X_train_compare, y_train_compare):
        super().updateGlobalModel(aggregated_weight_matrix, aggregated_bias_matrix, X_train_compare, y_train_compare)

    def updateLocalModels(self):
        super().updateLocalModels()

    def generateLoopMetrics(self, loop_number, X_test_compare, y_test_compare):
        super().generateLoopMetrics(loop_number, X_test_compare, y_test_compare)

    def printMetrics(self):
        super().printMetrics()

    def generateLocalModels(self, nb_clients):
        super().generateLocalModels(nb_clients)

    def aggregate(self, matrix):
        average = matrix[0]
        i = 1
        for elem in matrix[1:]:
            average += elem
            i += 1

        newList = [elem / i for elem in average]

        return newList

    def storeMetrics(self):
        super().storeMetrics()