from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump
import os

def gradientBoostingRegressor(train_set, energy_consumption):
    param_grid = [
        {'n_estimators': [10, 25, 50, 100], 'max_features': [5, 15, 25, 50, 75], 'learning_rate' : [0.1, 0.2], 'random_state' : [0, 25, 42, 60]}
    ]

    reg = GradientBoostingRegressor(random_state=0)

    grid_search = GridSearchCV(reg, param_grid, cv=10,
                           scoring='neg_mean_squared_error', return_train_score=True)
    print("\n Start training gradient boosting model ...")
    grid_search.fit(train_set, energy_consumption)
    print("\n Searching best estimator ...")
    best_reg = grid_search.best_estimator_
    print("\n Saving model ...")
    saveModel(best_reg)
    return best_reg

def saveModel(regressor):
    models_folder = os.path.join(os.path.dirname(__file__), "../_data/models")
    dump(regressor, models_folder + "/GradientBoosting.pkl")