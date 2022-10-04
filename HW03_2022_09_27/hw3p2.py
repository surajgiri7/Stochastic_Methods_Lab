#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: surajgiri
HW3 Problem 2
"""

from joblib import PrintTime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# changing date to only years
def change2yrs(YM):
    # if only month is given
    if "Y" not in YM:
        return int(YM[:-1])/12
    
    # if only year is given
    elif "M" not in YM:
        return int(YM[:-1])
    
    # if both year and month are given
    else:
        return int(YM.split("Y")[0]) + int(YM.split("Y")[1][:-1])/12

def preprocess(df, bondtype):
    # extracring the data of specific bond_type: G_N_C or G_N_A
    df = df[df['INSTRUMENT_FM'] == bondtype]
    # keeping the data that has sport rates
    df = df[df['DATA_TYPE_FM'].str.startswith('SR_')]
    df['DATA_TYPE_FM'] = df['DATA_TYPE_FM'].map(lambda x: str(x)[3:]).map(change2yrs)
    df = df.sort_values(by=['DATA_TYPE_FM'])
    return df

if __name__ == "__main__":
    df = pd.read_csv('data.csv', usecols=['DATA_TYPE_FM', 'OBS_VALUE', 'INSTRUMENT_FM'])
    df_GNA = preprocess(df, "G_N_A")
    df_GNC = preprocess(df, "G_N_C")

    plt.plot(df_GNA['DATA_TYPE_FM'], df_GNA['OBS_VALUE'], label='AAA rated bonds')
    plt.plot(df_GNA['DATA_TYPE_FM'], df_GNC['OBS_VALUE'], '--', label='All bonds')
    plt.legend()
    plt.title('Spots Rate vs. Years to Maturity')
    plt.xlabel('Residual maturity in years')
    plt.ylabel('Yield in %')
    plt.grid()
    plt.show()

