from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np

from datetime import datetime
import pandas as pd
import time
import matplotlib.pyplot as plt
import csv
from scipy import stats
from sklearn import preprocessing
import os
import traceback


def inspectData(data):
    checkNull = data.isnull().any()
    lengthData = len(data)
    maxVal = data.max()
    minVal = data.min()
    varVal = data.var()
    stdVal = data.std()
    meanVal = data.mean()
    medianVal = data.median()
    print('Data nullities: ', checkNull, ' | Data length: ', lengthData)
    print('Data max value: ', maxVal, ' | Data min value: ', minVal)
    print('Data variance: ', varVal, ' | Data standard deviation: ', stdVal)
    print('Data mean: ', meanVal, ' | Data median: ', medianVal)
    plotAny(data)
    
def plotAny(data):
    plt.figure(num=None, figsize=(24, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(data)
    plt.show()
    
def distCheck(x):
    mean = np.mean(x)
    std = np.std(x)
    print('mean: ', mean, ' std: ', std)
    
    snd = stats.norm(mean, std)
    # x = df['value']
    plt.figure(figsize=(7.5,7.5))
    plt.plot(x, snd.pdf(x))
    # plt.xlim(-60, 60)
    figTitle = 'Normal Distribution | Mean: ' + str(mean) + ' STD: ' + str(std)
    plt.title(figTitle, fontsize='15')
    plt.xlabel('Values of Variable X', fontsize='15')
    plt.ylabel('Probability', fontsize='15')
    plt.show()
    
def diff_tr(data, interval):
    transformed_tr = []
    for i in range(interval, len(data)):
        result = data[i] - data[i-interval]
        transformed_tr.append(result)
    return transformed_tr
    # return [data[i] - data[i - interval] for i in range(interval, len(data))]
    
def genLaggedFeat(df, n_lags):
    df_n = df.copy()
    df_nn = df.copy()
    for i in range(1, n_lags+1):
        df_a = pd.DataFrame()
        df_a[f"lag{i}"] = df_n.shift(i)
        df_nn = pd.concat([df_nn, df_a], axis=1, join='inner')
        # df_n[f"lag{i}"] = df_n["value"].shift(i)
    # df_n = df_n.iloc[n_lags:]
    
    return df_nn

def genTimeFeat(data):
    data = (data
                    .assign(minute = df_transformed.index.minute)
                    # .assign(day = df.index.day)
                    # .assign(month = df.index.month)
                    # .assign(day_of_week = df.index.day_of_week)
                    # .assign(week_of_year = df.index.week)
                    )
    return data

def genCylicalFeat(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}': lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}': lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)
    }
    return df.assign(**kwargs).drop(columns=col_name)


def conv_to_array(data):
    output = np.array(data)
    return output

def sliding_window(data, seq_length, labels_length):
    x = []
    y = []
    z = []
    
    for i in range(len(data)-(seq_length+labels_length)):
        _x = data.iloc[i:(i+seq_length),:]
        _y = data.iloc[(i+seq_length):(i+seq_length+labels_length),0:1]
        _z = data.iloc[(i+seq_length):(i+seq_length+labels_length),1:]
        x.append(np.array(_x))
        y.append(np.array(_y))
        z.append(np.array(_z))
        
    return x,y,z

# def create_dataset (X, y, look_back = 1):
#     Xs, ys = [], []
 
#     for i in range(0,len(X)-look_back):
#         v = X[i:i+look_back]
#         w = y[i+look_back]
#         Xs.append(v)
#         ys.append(w)
 
#     return np.array(Xs), np.array(ys)

def create_dataset(X, y, look_back=1, horizon=1):
    Xs, ys = [], []
    
    for i in range(len(X) - look_back - horizon + 1):
        Xs.append(X[i : i + look_back])
        ys.append(y[i + look_back : i + look_back + horizon])
    
    return np.array(Xs), np.array(ys)

def data_plot(df):
    # Plot line charts
    df_plot = df.copy()

    ncols = 2
    nrows = int(round(df_plot.shape[1] / ncols, 0))

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(14, 7))
    for i, ax in enumerate(fig.axes):
        sns.lineplot(data=df_plot.iloc[:, i], ax=ax)
        ax.tick_params(axis="x", rotation=30, labelsize=10, length=0)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.tight_layout()
    plt.show()