from calendar import day_abbr
from json import load
import os
from pathlib import Path

from wrangling.wrangling import load_data, analyze, drop_nan, sort_features

departments = [62]
# departments = [62, 83, 85, 91]
datasets = dict()

for department in departments:
    df = load_data(department)
    datasets[str(department)] = df

sort_features(datasets);

analyze(datasets)

drop_nan(datasets)

analyze(datasets)

# for department in datasets:
#     print(datasets[department].head())
#     csv_path = "./" + str(department) + ".csv"
#     datasets[department].to_csv(csv_path)
# for department in datasets:
#     print(datasets[str(department)].head())
    # print(datasets[department].where(datasets[department]['bnb_id'] == '620010000D0223_6a917ef55f897b7'))