from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump
import os

def stackingRegressor(train_set, energy_consumption):
    param_grid = [
        {'n_estimators': [10, 25, 50, 100], 'max_features': [5, 15, 25, 50, 75]}
    ]

    estimators = [
        ('lr', RidgeCV()),
        ('svr', LinearSVR(random_state=42))
    ]
    reg = StackingRegressor(
        estimators=estimators,
        final_estimator=RandomForestRegressor(n_estimators=10,
                                            random_state=42)
    )

    grid_search = GridSearchCV(reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
    print("\n Start training stacking model ...")
    grid_search.fit(train_set, energy_consumption)
    print("\n Searching best estimator ...")
    best_reg = grid_search.best_estimator_
    print("\n Saving model ...")
    saveModel(best_reg)
    return best_reg

def saveModel(regressor):
    models_folder = os.path.join(os.path.dirname(__file__), "../_data/models")
    dump(regressor, models_folder + "/Stacking.pkl")