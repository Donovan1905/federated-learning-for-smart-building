from calendar import day_abbr
from json import load
import os
from pathlib import Path
import pandas as pd
from matplotlib.pyplot import sca

from aggregation.FedAvgClass import FedAvgClass
from learning.main import performTraining
from aggregation.fedavg import federationLoop
from wrangling.wrangling import load_data, analyze, filling_nan, create_features, encode_to_num, scale



nb_loop = 5
testing_floor = 7
fedavg = FedAvgClass(nb_loop, 4, testing_floor)
