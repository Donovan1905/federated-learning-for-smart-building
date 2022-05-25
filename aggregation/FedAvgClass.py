from abc import ABC

import numpy as np

from AbstractFederated import AbstractFederated


class FedAvgClass(AbstractFederated, ABC):
    def __init__(self, nb_loop):
        print("FedAvg created")
        self.federationLoop(nb_loop)

    def federationLoop(self, nb_loop):
        super().federationLoop(nb_loop)

    def federate(self):
        print("------ FEDERATE LOCAL MODELS ------")
        weight_matrix = self.local_models[0].coefs_
        bias_matrix = self.local_models[0].intercepts_

        for model in self.local_models[1:]:
            weight_matrix += model.coefs_
            bias_matrix += model.intercepts_

        aggregated_weight_matrix = np.true_divide([2, 4, 6], 2).tolist()
        print("Aggregated weight matrix")
        print(aggregated_weight_matrix)

        aggregated_bias_matrix = np.true_divide(np.array(bias_matrix), len(self.local_models)).tolist()
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
