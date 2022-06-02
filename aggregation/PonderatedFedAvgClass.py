from abc import ABC

from aggregation.AbstractFederated import AbstractFederated


class PonderatedFedAvgClass(AbstractFederated, ABC):
    def __init__(self, nb_loop):
        print("PonderatedFedAvg created")
        self.federationLoop(nb_loop)

    def federationLoop(self, nb_loop):
        super().federationLoop(nb_loop)

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

    def updateGlobalModel(self, aggregated_weight_matrix, aggregated_bias_matrix):
        super().updateGlobalModel(aggregated_weight_matrix, aggregated_bias_matrix)

    def updateLocalModels(self):
        super().updateLocalModels()

    def generateLoopMetrics(self, loop_number):
        super().generateLoopMetrics(loop_number)

    def printMetrics(self):
        super().printMetrics()

    def generateLocalModels(self):
        super().generateLocalModels()

    def aggregate(self, matrix):
        average = matrix[0]
        i = 1
        for elem in matrix[1:]:
            average += elem
            i += 1

        newList = [elem / i for elem in average]

        return newList