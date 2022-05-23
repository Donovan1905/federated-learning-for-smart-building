from sklearn.linear_model import LinearRegression
from joblib import dump
import os

def linearRegressor(train_set, energy_consumption):
    reg = LinearRegression()

    print("\n Start training linear model ...")
    reg.fit(train_set, energy_consumption)
    print("\n Saving model ...")
    saveModel(reg)
    return reg

def saveModel(regressor):
    models_folder = os.path.join(os.path.dirname(__file__), "../_data/models")
    dump(regressor, models_folder + "/Linear.pkl")