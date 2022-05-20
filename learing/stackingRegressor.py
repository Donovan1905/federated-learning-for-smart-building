from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import GridSearchCV

def stackingRegressor(x_train, y_train):
    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
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
    best_reg = grid_search.best_estimator_
    
    best_reg.fit(x_train, y_train)
    return best_reg