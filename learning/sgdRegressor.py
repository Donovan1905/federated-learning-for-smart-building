from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump
import os

def sgdRegressor(train_set, energy_consumption):
    param_grid = [
        {'max_iter': [2000, 4000, 6000, 8000, 10000], 'alpha' : [0.0001, 0.0002], 'random_state' : [0, 25, 42, 60]}
    ]

    reg = SGDRegressor(max_iter=1000, tol=1e-3)
    grid_search = GridSearchCV(reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
    print("\n Start training sgd model ...")
    grid_search.fit(train_set, energy_consumption)
    print("\n Searching best estimator ...")
    best_reg = grid_search.best_estimator_
    print("\n Saving model ...")
    saveModel(best_reg)
    return best_reg

def saveModel(regressor):
    models_folder = os.path.join(os.path.dirname(__file__), "../_data/models")
    dump(regressor, models_folder + "/SGD.pkl")