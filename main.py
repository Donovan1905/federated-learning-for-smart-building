from calendar import day_abbr
from json import load
import os
from pathlib import Path

from wrangling.wrangling import load_data, analyze, filling_nan, sort_features, encode_to_num

departments = [62]
# departments = [62, 83, 85, 91]
datasets = dict()

for department in departments:
    df = load_data(department)
    datasets[str(department)] = df

sort_features(datasets);

analyze(datasets)

filling_nan(datasets)

encode_to_num(datasets)

analyze(datasets)

for department in datasets:
    print(datasets[str(department)].head())

# for department in datasets:
#     print(datasets[department].head())
#     csv_path = "./" + str(department) + ".csv"
#     datasets[department].to_csv(csv_path)
# for department in datasets:
#     print(datasets[str(department)].head())
    # print(datasets[department].where(datasets[department]['bnb_id'] == '620010000D0223_6a917ef55f897b7'))