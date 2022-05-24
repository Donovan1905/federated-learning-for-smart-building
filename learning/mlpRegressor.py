from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump
import os

def mlpRegressor(train_set, energy_consumption):

    reg = MLPRegressor(random_state=1, max_iter=10000)

    print("\n Start training neuronal network model ...")
    reg.fit(train_set, energy_consumption)
    print("\n Saving model ...")
    saveModel(reg)
    return reg

def saveModel(regressor):
    models_folder = os.path.join(os.path.dirname(__file__), "../_data/models")
    dump(regressor, models_folder + "/MLP.pkl")
