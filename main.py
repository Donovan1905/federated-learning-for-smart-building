from calendar import day_abbr
from json import load
import os
from pathlib import Path

from matplotlib.pyplot import sca

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
