from multiprocessing.sharedctypes import Value
import os 
import pandas as pd
import numpy as np
from pyparsing import col
from .params import *
import pprint
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys
from datetime import date, datetime


oe = preprocessing.OneHotEncoder()

def load_data(floor):
    csv_path_2018 = os.path.join(os.path.dirname(os.path.abspath(__file__)), ('../_data/csv/2018Floor' + str(floor) + '.csv'))
    csv_path_2019 = os.path.join(os.path.dirname(os.path.abspath(__file__)), ('../_data/csv/2019Floor' + str(floor) + '.csv'))
    print("Loading 2018... ", csv_path_2018)
    csv_2018 = pd.read_csv(csv_path_2018, sep=',', low_memory=False)
    print("Loading 2019... ", csv_path_2019)
    csv_2019 = pd.read_csv(csv_path_2019, sep=',', low_memory=False)

    print("Merging...")
    csv = pd.concat([csv_2018, csv_2019])
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

def filling_nan(dict):
    print("\nFilling NaN values... ")
    for floor in dict:
        columns_with_nan = checkColumnsWithNan(dict[floor])

        for column in columns_with_nan: 
            print("\nFilling ", column, " with mean")
            dict[floor][column].fillna(round(dict[floor][column].mean()), inplace=True)
    return dict

def create_features(dict):
    print("Calculate energy consumption...")
    for floor in dict:      
        tot_energy = [] 
        for index, row in dict[str(floor)].iterrows():
            energy_consumption = 0
            for column in energy_consumptions:
                energy_consumption += int(row[column])
            tot_energy.append(energy_consumption)
            sys.stdout.write("%d out of %d with %d kW\r" % (index, len(dict[floor]), energy_consumption))
            sys.stdout.flush()
        dict[floor]['tot_energy'] = tot_energy

def encode_to_num(dict):
    print("Convert non-numerical values...")
    for floor in dict:
        unix_epoch = []
        for i in range(0, len(dict[floor])):
            try:
                date = dict[floor].loc[i, 'Date']
                if (type(date) == str):
                    unix_date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').timestamp()
                else:
                    unix_date = datetime.strptime(date.values[0], '%Y-%m-%d %H:%M:%S').timestamp()
                # date = datetime.strptime(dict[floor].loc[i, 'Date'].values[0], '%Y-%m-%d %H:%M:%S').timestamp()
                unix_epoch.append(unix_date)
            except KeyError:
                unix_epoch.append(np.nan)


            
            sys.stdout.write("%d out of %d \r" % (i, len(dict[floor])))
            sys.stdout.flush()
        dict[floor].drop(['Date'], axis=1)
        dict[floor]['timestamp'] = unix_epoch
        dict[floor].dropna(inplace=True)

def scale(dict):
    for floor in dict:
        del dict[floor]['Date']
        scaler = preprocessing.StandardScaler()
        scaled = scaler.fit_transform(dict[floor])
        dict[floor] = pd.DataFrame(scaled, columns=dict[floor].columns)

def checkColumnsWithNan(df):
    columnsWithNaN = dict()
    for column in df:
        isNaN = (df[column].isna().sum())
        if isNaN != 0:
            columnsWithNaN[column] = isNaN
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(columnsWithNaN)
    return columnsWithNaN
