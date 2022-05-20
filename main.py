from calendar import day_abbr
from json import load
import os
from pathlib import Path

from wrangling.wrangling import load_data, analyze, drop_nan

departments = [62, 83, 85, 91]
datasets = dict()

for department in departments:
    df = load_data(department)
    datasets[str(department)] = df

# for department in datasets:
#     print(datasets[department].head())

analyze(datasets)

_data = drop_nan(datasets)

analyze(_data)

