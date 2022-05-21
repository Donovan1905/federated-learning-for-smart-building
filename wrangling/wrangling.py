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
    print("\nAnalyzing...")
    for department in dict:
        print("\n-------------------------------------------------Department number : ", str(department), "--------------------------------------------------------------------\n")
        nb_features = len(dict[str(department)].columns.values)
        columns_with_nan = checkColumnsWithNan(dict[department])
        nb_nan_row = dict[department].isna().sum().sum()
        nb_line = len(dict[department].index)

        print("Number of line : ", str(nb_line))
        print("\nNumber of columns with NaN values : ", str(len(columns_with_nan)), " and number of row with NaN values :", str(nb_nan_row))
        print('\nNumber of feature : ',str(nb_features))
        print("\n", dict[department].describe())

def sort_features(dict):
    print("\nSorting features... \n")
    for department in dict:
        for column in dict[department]:
            if(column not in features):
                del (dict[department])[column]
                print("Deleting", column, " in ", str(department))

def drop_nan(dict):
    print("\nRemoving NaN values... ")
    for department in dict:
        columns_with_nan = checkColumnsWithNan(dict[department])
        dict[department].dropna(subset=columns_with_nan, inplace=True)
    print(columns_with_nan)
    return dict

def checkColumnsWithNan(df):
    columnsWithNaN = []
    for column in df:
        isNaN = (df[column].isna().sum())
        if isNaN != 0:
            columnsWithNaN.append(column)
    return columnsWithNaN
