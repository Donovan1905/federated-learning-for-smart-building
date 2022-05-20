from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def gradientBoostingRegressor(x_train, y_train):
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]

    reg = GradientBoostingRegressor(random_state=0)

    grid_search = GridSearchCV(reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
    best_reg = grid_search.best_estimator_
    
    best_reg.fit(x_train, y_train)
    return best_reg