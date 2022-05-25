from calendar import day_abbr
from json import load
import os
from pathlib import Path
import pandas as pd
from matplotlib.pyplot import sca
from learning.main import performTraining
from aggregation.fedavg import federationLoop
from wrangling.wrangling import load_data, analyze, filling_nan, create_features, encode_to_num, scale

floors = [3]
# floors = [3, 4 ,5, 6, 7]
datasets = dict()

for floor in floors:
    df = load_data(floor)
    datasets[str(floor)] = df

filling_nan(datasets)
create_features(datasets)
encode_to_num(datasets)
scale(datasets)

for floor in datasets:
    print(datasets[str(floor)].head())

for floor in datasets:
    datasets[floor].to_csv(("./_data/csv/wrangled-floor-" + str(floor) + ".csv"))

dataset_folder = os.path.join(os.path.dirname(__file__), "./_data/csv")
dataset = pd.read_csv(dataset_folder + "/wrangled-floor-3.csv")
performTraining(dataset)

#nb_loop = 5
#federationLoop(nb_loop)
