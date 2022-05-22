from learing.baggingRegressor import baggingRegressor
from learing.gradientBoostingRegressor import gradientBoostingRegressor
from learing.sgdRegressor import sgdRegressor
from learing.randomForestRegressor import rfRegressor
from learing.stackingRegressor import stackingRegressor
from learing.gradientBoostingRegressor import gradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def performTraining(dataset):
    train_set, energy_consumption, test_set = generateTrainingData(dataset)
    models_list = createModels(train_set, energy_consumption)
    models_comparison = compareModels(train_set, energy_consumption, models_list)

def generateTrainingData(dataset):
    train_set, test_set = split_train_test(dataset, 0.2)

    train_set = train_set.drop("nom_de_la_colonne_à_predict", axis=1)
    energy_consumption = train_set["nom_de_la_colonne_à_predict"].copy()

    return train_set, energy_consumption, test_set

def createModels(train_set, energy_consumption):
    models_list = []

    models_list.append(baggingRegressor(train_set, energy_consumption))
    models_list.append(gradientBoostingRegressor(train_set, energy_consumption))
    models_list.append(sgdRegressor(train_set, energy_consumption))
    models_list.append(rfRegressor(train_set, energy_consumption))
    models_list.append(stackingRegressor(train_set, energy_consumption))
    models_list.append(gradientBoostingRegressor(train_set, energy_consumption))

    return models_list

def compareModels(train_set, energy_consumption, models_list):
    models_results = []

    for model in models_list:
        scores = cross_val_score(model, train_set, energy_consumption, scoring="neg_mean_squared_error", cv=10)
        
        model_scores = np.sqrt(-scores)
        results = [model_scores.mean(), model_scores.std()]

        models_results.append(results)

    return models_results

