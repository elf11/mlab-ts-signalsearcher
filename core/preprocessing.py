from __future__ import print_function
import pandas as pd 
import numpy as np 
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import sys

import os

rcParams['figure.figsize'] = 15,6

#dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
#data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month',date_parser=dateparse)
#print data.head()
#print '\n Data Types:'
#print data.dtypes

#sts = data['#Passengers']
#print sts.head(10)

#plt.plot(ts)
#plt.show()


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def read_cached(name):
    """
    Reads csv file (maybe zipped) from data directory and caches it's content as a pickled DataFrame
    :param name: file name without extension
    :return: file content
    """
    cached = 'data/%s.pkl' % name
    sources = ['data/%s.csv' % name, 'data/%s.csv.zip' % name]
    if os.path.exists(cached):
        return pd.read_pickle(cached)
    else:
        for src in sources:
            if os.path.exists(src):
                df = pd.read_csv(src)
                vc = df.key.value_counts()
                df = df[df.key.isin(vc.index[vc.values > 100])]
                dwn = df[['key', 'date', 'download']].copy()
                #dwn.set_index('key', inplace=True)
                up = df[['key', 'date', 'upload']].copy()
                #up.set_index('key', inplace=True)
                rtt = df[['key', 'date', 'rtt']].copy()
                #rtt.set_index('key', inplace=True)
                dwn.to_pickle('dataFile/download.pkl')
                up.to_pickle('dataFile/upload.pkl')
                rtt.to_pickle('dataFile/rtt.pkl')
                df.to_pickle(cached)
                return df


def read_all():
    """
    Reads source data for training/prediction
    """
    def read_file(file):
        df = read_cached(file).set_index('key')
        #print df.groupby(df.index.get_level_values(0)).count()
        #df = df.drop(df[(df.groupby(df.index.get_level_values(0)).count()) < 50].index)
        #df.drop(df[df.groupby(df.index.get_level_values(0)).count() < 50].index, inplace=True)
        #print df.count()
        return df

    # Path to cached data
    path = os.path.join('data', 'all.pkl')
    if os.path.exists(path):
        df = pd.read_pickle(path)
    else:
        # Official data
        df = read_file('country')

        df = df.sort_index()
        # Cache result
        df.to_pickle(path)
    return df


def pivot_table(name):
    toRead = '%s.pkl' % name
    path = os.path.join('/home/oniculaescu/mlab-ts-signalsearcher/dataFile', toRead)
    ret = pd.read_pickle(path)
    ret = ret.groupby(['key', 'date'], as_index=False)[name].mean()
    ret = ret.pivot(index='key', columns='date', values=name)
    ret = ret.rename_axis(None)
    
    return ret

data = pivot_table('download')
data.index.rename('asname', inplace=True)
#eprint(data.index.values, sys.stdout)
# count the number of non NaN values in each row
#eprint(data.apply(lambda x: x.count(), axis=1), sys.stdout)
# select only those rows that have more than 2900 values presented/non NaN
data_res = data[data.apply(lambda x: x.count(), axis=1) >= 2900]
#eprint(data_res.index, sys.stdout)

path = os.path.join('/home/oniculaescu/mlab-ts-signalsearcher/dataFile/download.pkl')
ret = pd.read_pickle(path)
ret2 = ret.copy()
ret2.set_index('key', inplace=True)
#eprint(ret.index, sys.stdout)
#eprint(ret, sys.stdout)
header = ['date', 'download']
for i in data_res.index:
    if i in ret2.index:
        eprint(i, sys.stdout)
        ret_red = ret.loc[ret['key'] == i]
        #eprint(ret_red, sys.stdout)
        name = i + '.csv'
        ret_red.to_csv(name, columns=header, sep=',', encoding='utf-8', index=False)

#data_res.set_index('asname', inplace=True)
#eprint(data_res.columns, sys.stdout)
#eprint(data_res, sys.stdout)
#data_res.to_csv('atMost10procent.csv', sep=',', encoding='utf-8', index=False)
#df = pd.read_csv('atMost10procent.csv')
#eprint(df, sys.stdout)

'''
for i in df.index:
    df_n = df[i:]
    eprint(df_n, sys.stdout)
    df_n = df_n.pivot()
    name = str(i) + '.csv'
    df_n.to_csv(name, sep=',', encoding='utf-8')
'''

