from calendar import day_abbr
from json import load
import os
from pathlib import Path

from matplotlib.pyplot import sca

from wrangling.wrangling import load_data, analyze, filling_nan, create_features, encode_to_num, scale, create_global_df

floors = [3, 4 ,5, 6, 7]
all_floors = [3, 4 ,5, 6, 7]
datasets = dict()

wrangling_state = dict()
for floor in floors:
    df = load_data(floor)
    datasets[str(floor)] = df['csv']
    wrangling_state[str(floor)] = df['wrangled']




filling_nan(datasets, wrangling_state)

create_features(datasets, wrangling_state)
encode_to_num(datasets, wrangling_state)
scale(datasets, wrangling_state)

global_data = create_global_df(datasets)
print("Merged all datasets into a global with ", len(global_data.index), " lines")


for floor in datasets:
    if(wrangling_state[floor] == False):
        print("Save csv for floor ", str(floor))
        datasets[floor].to_csv(("./_data/csv/wrangled-floor-" + str(floor) + ".csv"))
    else: 
        print("Skip csv save for floor ", str(floor))