import os 
import pandas as pd
import numpy as np
from .params import *
import pprint
import matplotlib.pyplot as plt
from sklearn import preprocessing
oe = preprocessing.OneHotEncoder()


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

def filling_nan(dict):
    print("\nFilling NaN values... ")
    for department in dict:
        columns_with_nan = checkColumnsWithNan(dict[department])

        for column in columns_with_nan: 
            if(column in num_features):
                print("\nFilling ", column, " with mean")
                dict[department][column].fillna(round(dict[department][column].mean()), inplace=True)
            
            if(column in enum_features):
                print("\nFilling ", column, " with most frequent occurence")
                most_frequent_occ = dict[department][column].value_counts().index[0]
                dict[department][column].fillna(most_frequent_occ, inplace=True)
    return dict

def encode_to_num(dict):
    for department in dict:
        dict[department] = pd.get_dummies(dict[department], columns=enum_features, drop_first=True)

def scale(dict):
    for department in dict:
        scaler = preprocessing.StandardScaler()
        scaled = scaler.fit_transform(dict[department])
        dict[department] = pd.DataFrame(scaled, columns=dict[department].columns)

def checkColumnsWithNan(df):
    columnsWithNaN = dict()
    for column in df:
        isNaN = (df[column].isna().sum())
        if isNaN != 0:
            columnsWithNaN[column] = isNaN
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(columnsWithNaN)
    return columnsWithNaN
