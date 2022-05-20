import csv
import os 
import pandas as pd
import numpy as np

def load_data(department):
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../_data/csv/', str(department) + '/batiment.csv')
    print("Loading... ", csv_path)
    csv = pd.read_csv(csv_path, sep=';', low_memory=False)
    return csv

def analyze(dict):
    print("Analyzing...")
    for department in dict:
        nb_features = len(dict[str(department)].columns.values)
        
        
        print('Table "batiment" from department : ', str(department), " have ", str(nb_features), " different features\n")
        
