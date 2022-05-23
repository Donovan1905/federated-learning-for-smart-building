from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV

def baggingRegressor(train_set, energy_consumption):
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    reg = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0)
    
    grid_search = GridSearchCV(reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
    best_reg = grid_search.best_estimator_
    
    best_reg.fit(train_set, energy_consumption)
    return best_reg