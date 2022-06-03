from calendar import day_abbr
from json import load
import os
from pathlib import Path
import pandas as pd
from matplotlib.pyplot import sca
from learning.main import performTraining
from aggregation.fedAvg import fedAvg
from aggregation.fedDist import fedDist
from wrangling.wrangling import load_data, analyze, filling_nan, create_features, encode_to_num, scale, create_global_df
import numpy as np
import glob

# nb_loop = 5
# testing_floor = 7

floors = [3, 4, 5, 6, 7]
local_clients = [3, 4, 5, 6]
testing_floor = 7
dataset_folder = os.path.join(os.path.dirname(__file__), "./_data/csv")
datasets = dict()

wrangling_state = dict()
for floor in floors:
    df = load_data(floor)
    datasets[str(floor)] = df['csv']
    wrangling_state[str(floor)] = df['wrangled']

filling_nan(datasets, wrangling_state)

create_features(datasets, wrangling_state)
encode_to_num(datasets, wrangling_state)
# scale(datasets, wrangling_state)

global_data = create_global_df(datasets)
print("Merged all datasets into a global with ", len(global_data.index), " lines")

for floor in datasets:
    if(wrangling_state[floor] == False):
        print("Save csv for floor ", str(floor))
        datasets[floor].to_csv(("./_data/csv/wrangled-floor-" + str(floor) + ".csv"))
    else: 
        print("Skip csv save for floor ", str(floor))

# for client in local_clients:
#     performTraining(client, testing_floor)
nb_loop = 5
print("----- START RUNNING FED_AVG ALGORITHM -----")
fedAvg(nb_loop, len(local_clients), testing_floor)
print("----- START RUNNING FED_DIST ALGORITHM -----")
fedDist(nb_loop, len(local_clients), testing_floor)

# fedavg = FedAvgClass(nb_loop, 4, testing_floor)
