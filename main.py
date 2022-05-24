from calendar import day_abbr
from json import load
import os
from pathlib import Path

from matplotlib.pyplot import sca

from wrangling.wrangling import load_data, analyze, filling_nan, sort_features, encode_to_num, scale
from learning.main import performTraining
from aggregation.fedavg import federationLoop

departments = [62]
#departments = [62, 83, 85, 91]
datasets = dict()

#for department in departments:
   # df = load_data(department)
   # datasets[str(department)] = df

#analyze(datasets)
#sort_features(datasets)
#filling_nan(datasets)
#encode_to_num(datasets)
#scale(datasets)
#analyze(datasets)

#for department in datasets:
#    results = performTraining(datasets[str(department)])
#    print(results)

#performTraining()

nb_loop = 10
federationLoop(nb_loop)

