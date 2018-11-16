import pandas as pd 
import numpy as np 
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

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
    path = os.path.join('/home/oniculaescu/lstm/data', toRead)
    ret = pd.read_pickle(path)
    ret = ret.groupby(['key', 'date'], as_index=False)['download'].mean()
    ret = ret.pivot(index='key', columns='date', values=name)
    ret = ret.rename_axis(None)
    
    return ret

