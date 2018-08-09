import pandas as pd 
import numpy as np 

import numba

import stationary as st 


def read_start_end(attr, start, end):
    """
    Return source data from start date to end date for the attribute in attr
    attr can be [dwn, up, rtt]
    """
    df = st.pivot_table(attr)
    if start and end:
        return df.loc[:, start:end]
    elif end:
        return df.loc[:, :end]
    else:
        return df

@numba.jit(nopython=True)
def single_autocorrelation(series, lag):
    """
    Autocorrelation for single data series
    :param series: traffic series
    :param lag: lag, days
    :return:
    """

    s1 = series[lag:]
    s2 = series[:-lag]
    ms1 = np.mean(s1)
    ms2 = np.mean(s2)
    ds1 = s1 - ms1
    ds2 = s2 - ms2

    divider = np.sqrt(np.sum(ds1 * ds1)) * np.sqrt(np.sum(ds2 * ds2))
    return np.sum(ds1 * ds2) / divider if divider != 0 else 0

@numba.jit(nopython=True)
def find_start_end_series(data):
    """
    Calculate the start and end dates for the time series
    start date = index of first non zero non-NAN value
    end date = index of last non zero non-NAN value
    :param data: Time series, shape [n_astimeseries, n_days]
    """

    n_astimeseries = data.shape[0]
    n_days = data.shape[1]
    startidx = np.full(n_astimeseries, -1, dtype=np.int32)
    endidx = np.full(n_astimeseries, -1, dtype=np.int32)

    for asseries in range(n_astimeseries):
        # scan from start to end
        for day in range(n_days):
            if not np.isnan(data[asseries, day]) and data[asseries, day] > 0:
                startidx[asseries] = day
                break
    
        #reverse the scan for finding end date, scan from end to start
        for day in range(n_days):
            if not np.isnan(data[asseries, day]) and data[asseries, day] > 0:
                endidx[asseries] = day
                break
    
    return startidx, endidx

def prepare_data(attr, start, end, threshold):
    """
    Reads the data source, and calculates the start and end dates for each of the series, calculates log1p(series)
    :param start: start date of effective time interval, if None = start from beginning
    :param end: end date of effective time interval, if None = return all data
    :param threshold: minimal ratio of series real length to entire (end-start) interval. Drop the series if ratio is less than threshold
    :return: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray] = tuple(log1p(series), nans, series start, series end)
    """
    df = read_start_end(attr, start, end)
    starts, ends = find_start_end_series(df.values)
    # create a boolean mask for eliminating series that are too short
    mask = (ends - starts) / df.shape[1] < threshold
    inv_mask = ~mask
    df = df[inv_mask]
    nans = pd.isnull(df)
    return np.log1p(df.fillna(0)), nans, starts[inv_mask], ends[inv_mask]

def lag_indexes(start, end):
    """
    Calculate indexes for 3,6,9.12 months backwards lag for the given data
    :param start: start of range
    :param end: end of range
    :return: list of 4 series, one for each lag. List[pd.Series]
    if the target date backward is out of the (start, end) range then index is -1
    """
    range = pd.date_range(start, end)
    #key is date, value is day index
    base_index = pd.Series(np.range(0, len(range)), index=range)

    def lag(offset):
        dates = range - offset
        return pd.Series(data=base_index.loc[dates].fillna(-1).astype(np.int16).values, index=range)
    
    return [lag(pd.DateOffset(months=m)) for m in (3,6,9,12)]

