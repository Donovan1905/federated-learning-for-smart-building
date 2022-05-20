import csv
import os 
import pandas as pd
import numpy as np
from .params import *


def load_data(department):
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../_data/csv/', str(department) + '/batiment.csv')
    print("Loading... ", csv_path)
    csv = pd.read_csv(csv_path, sep=';', low_memory=False)
    return csv

def analyze(dict):
    print("Analyzing...")
    for department in dict:
        nb_features = len(dict[str(department)].columns.values)
        columns_with_nan = checkColumnsWithNan(dict[department])
        nb_line = dict[department].all().sum()

        print("Department ", str(department), " have ", str(nb_line), " lines\n")
        print("Departement ", str(department), " have NaN in : ", str(len(columns_with_nan)), "\n")
        print('Table "batiment" from department : ', str(department), " have ", str(nb_features), " different features\n")

def sort_features(dict):
    print("Sorting features... \n")
    for department in dict:
        for column in dict[department]:
            if(column not in features):
                del (dict[department])[column]
                print("Deleting", column, " in ", str(department))

def drop_nan(dict):
    print("Removing NaN values... ")
    for department in dict:
        columns_with_nan = checkColumnsWithNan(dict[department])
        for column in columns_with_nan:
            dict[department].dropna(inplace=True)
    return dict

def checkColumnsWithNan(df):
    columnsWithNaN = []
    for column in df:
        isNaN = (df[column].isna().sum())
        if isNaN != 0:
            columnsWithNaN.append([column, df[column].isna().sum()])
    return columnsWithNaN