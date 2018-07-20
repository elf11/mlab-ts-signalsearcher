import pandas as pd 
import numpy as np 
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

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
    cached = 'dataFile/%s.pkl' % name
    sources = ['dataFile/%s.csv' % name, 'dataFile/%s.csv.zip' % name]
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
    path = os.path.join('dataFile', 'all.pkl')
    if os.path.exists(path):
        df = pd.read_pickle(path)
    else:
        # Official data
        df = read_file('july2016dec2017')

        df = df.sort_index()
        # Cache result
        df.to_pickle(path)
    return df


def pivot_table(name):
    toRead = '%s.pkl' % name
    path = os.path.join('/Users/oniculaescu/mlab/dataFile', toRead)
    ret = pd.read_pickle(path)
    ret = ret.groupby(['key', 'date'], as_index=False)['download'].mean()
    ret = ret.pivot(index='key', columns='date', values=name)
    ret = ret.rename_axis(None)
    
    return ret


def test_stationarity(timeseries):
    # determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #rolmean = pd.rolling_mean(timeseries, window=12)
    #rolstd = pd.rolling_std(timeseries, window=12)

    #plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()


    #Perform Dickey-Fuller test
    print 'Results of Dickey-Fuller test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critival Value (%s)'%key] = value
    print dfoutput

#test_stationarity(ts)

def transform_stationary(timeseries):
    ts_log = np.log(timeseries)
    #print 'here'
    #print ts_log.dtype
    ts_log = np.nan_to_num(ts_log)
    #print ts_log.head()
    # take weigthed moving average to make it stationary
    decomposition = seasonal_decompose(ts_log, freq=30)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    #ts_log_decompose = residual
    #ts_log_decompose.dropna(inplace=True)
    #test_stationarity(ts_log_decompose)

#transform_stationary(s)