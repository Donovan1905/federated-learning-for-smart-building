import csv
import os 
import pandas as pd
import numpy as np

# pd.set_option('display.width', 40)

def load_data(department):
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../_data/csv/', str(department) + '/batiment.csv')
    print(csv_path)
    csv = pd.read_csv(csv_path, sep=';')
    print(csv.head())
