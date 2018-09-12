import pandas as pd 
import numpy as np 

import numba

import argparse

import preprocessing as st 

from feeder import VarFeeder

import datetime

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
def batch_autocorrelation(data, lag, starts, ends, threshold, backoffset=0):
    """
    Autocorrelation for all timeseries at once
    :param data: timeseries, shape [n_astimeseries, n_days]
    :param lag: lag, days
    :param starts: vector of start index for each series
    :param ends: vector of end index for each series
    :param threshold: ratio of timeseries to lag to calculate autocorrelation
    :param backoffset: offset from the series end, in days
    :return: autocorrelation, shape [n_series]. if series is to short then autocorrelation is NaN
    """
    n_series = data.shape[0]
    n_days = data.shape[1]
    max_end = n_days - backoffset
    corr = np.empty(n_series, dtype=np.float64)
    support = np.empty(n_series, dtype=np.float64)
    for i in range(n_series):
        series = data[i]
        end = min(max_end, ends[i])
        real_len = end - starts[i]
        support[i] = real_len / lag
        if support[i] > threshold:
            series = series[starts[i]:end]
            # average lag between exact lag and two nearest neighbors for smoothnes
            c_365 = single_autocorrelation(series, lag)
            c_364 = single_autocorrelation(series, lag - 1)
            c_366 = single_autocorrelation(series, lag + 1)
            corr[i] = 0.5 * c_365 + 0.25 * c_364 + 0.25 * c_366
        else:
            corr[i] = np.NaN
    
    return corr



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
    ran = pd.date_range(start, end)
    #key is date, value is day index
    base_index = pd.Series(np.arange(0, len(ran)), index=ran)

    def lag(offset):
        dates = ran - offset
        return pd.Series(data=base_index.loc[dates].fillna(-1).astype(np.int16).values, index=ran)
    
    return [lag(pd.DateOffset(months=m)) for m in (3,6,9,12)]

def normalize(values):
    return (values - values.mean()) / np.std(values)

def read_countries_dict():
    countries = {}
    val = 0
    with open('countries.csv') as f:
        for line in f:
            var = "".join(line.split())
            countries[var] = val
            val = val + 1
    return countries


def uniq_country_map(ases):
    """
    Find AS country for all unique ASes i.e. group ASes by country
    :param ases: all ASes (must be presorted)
    :return: array[num_unique_ases, num_countries_world], where each column corresponds to country and each row corresponds to unique AS
    Value is an index of AS in source ASes array, if country is missing, value is -1
    """
    countries = read_countries_dict()
    lenC = len(countries)
    result = np.full([len(ases), lenC], -1, dtype=np.int32)

    #for key, value in countries.iteritems():
    #    print key, value

    prev_as = None
    num_page = -1
    for i, entity in enumerate(ases):
        key_as, loc = entity.rsplit('|', 1)
        country = loc[2:4]
        if entity != prev_as:
            prev_as = entity
            num_page += 1
        #print country
        #print countries[country]
        result[num_page, countries[country]] = i
    return result[:num_page+1]

def uniq_continent_map(ases):
    """
    NOTE TODO -already done just make sure you see it: for this one you can only send the keys in, you extract country/continent from the key
    Find AS continent for all unique ASes, i.e. group ASes by continent
    :param asses: all ASes (must be presorted)
    :return: array[num_unique_ases, 7], where each column corresponds to continent and each row corresponds to unique AS
    Value is an index of AS in source ASes array, if continent is missing, value is -1
    """
    result = np.full([len(ases), 7], -1, dtype=np.int32)

    prev_as = None
    num_page = -1
    continents = {'af' : 0, 'na' : 1, 'oc' : 2, 'an' : 3, 'as' : 4, 'eu' : 5, 'sa' : 6}
    for i, entity in enumerate(ases):
        key_as, loc = entity.rsplit('|', 1)
        cont = loc[:2]
        if entity != prev_as:
            prev_as = entity
            num_page += 1
        result[num_page, continents[cont]] = i
    return result[:num_page+1]


def run():
    parser = argparse.ArgumentParser(description='Prepare data')
    parser.add_argument('--data_dir',
                        default='/home/oniculaescu/train/features',
                        help='The location where we are going to save the features for constructing the NN')
    parser.add_argument('--threshold', 
                        default=0.0, 
                        type=float, 
                        help='Series minimal length threshold/points of data length')
    parser.add_argument('--add_days', 
                        default=64, 
                        type=int, 
                        help='Number of days to be added in the future for prediction')
    parser.add_argument('--start', help='Effective start date')
    parser.add_argument('--end', help="Effective end date")
    parser.add_argument('--attr', default='download', help="tell what pkl file to use for feature creation")
    parser.add_argument('--corr_backoffset', 
                        default=0, 
                        type=int, 
                        help="Offset for correlation computation")
    
    args = parser.parse_args()

    # get the data
    df, nans, starts, ends = prepare_data(args.attr, args.start, args.end, args.threshold)

    # find the working date range
    data_start, data_end = df.columns[0], df.columns[-1]

    # project date-dependent features (like day of week) to the future dates for prediction
    #features_end = data_end + " " + str(pd.Timedelta(args.add_days, unit='D'))
    print data_end
    data_end_1 = datetime.datetime.strptime(data_end, '%Y-%m-%d')
    #features_end = data_end_1 + pd.TimeDelta(args.add_days, unit='D')
    features_end = data_end_1 + datetime.timedelta(days=args.add_days)
    print("start: " + data_start + ", end: " + data_end + ", features_end: " + str(features_end))

    # Group unique ases by continent
    assert df.index.is_monotonic_increasing
    continent_map = uniq_continent_map(df.index.values)

    # Group unique ases by country
    country_map = uniq_country_map(df.index.values)

    # yearly autocorrelation
    raw_year_autocorr = batch_autocorrelation(df.values, 365, starts, ends, 1.5, args.corr_backoffset)
    year_unknown_pct = np.sum(np.isnan(raw_year_autocorr))/len(raw_year_autocorr) # type: float

    # quarterly autocorrelation
    raw_quarter_autocorr = batch_autocorrelation(df.values, int(round(365.25/4)), starts, ends, 2, args.corr_backoffset)
    quarter_unknown_pct = np.sum(np.isnan(raw_quarter_autocorr)) / len(raw_quarter_autocorr) # type: float
    
    print("Percent of undefined autocorr = yearly:%.3f, quarterly:%.3f" % (year_unknown_pct, quarter_unknown_pct))
    
    # Normalise all the things
    year_autocorr = normalize(np.nan_to_num(raw_year_autocorr))
    quarter_autocorr = normalize(np.nan_to_num(raw_quarter_autocorr))

    # Make time-dependent features
    features_days = pd.date_range(data_start, features_end)
    #dow = normalize(features_days.dayofweek.values)
    week_period = 7 / (2 * np.pi)
    dow_norm = features_days.dayofweek / week_period
    print dow_norm
    dow = np.stack([np.cos(dow_norm), np.sin(dow_norm)], axis=-1)

    # Assemble indices for quarterly lagged data
    lagged_ix = np.stack(lag_indexes(data_start, features_end), axis=-1)

    #page_popularity = df.median(axis=1)
    #page_popularity = (page_popularity - page_popularity.mean()) / page_popularity.std()

    # Put NaNs back
    df[nans] = np.NaN

    # Assemble final output
    tensors = dict(
        hits=df,
        lagged_ix=lagged_ix,
        continent_map=continent_map,
        #country_map=country_map,
        as_ix=df.index.values,
        year_autocorr=year_autocorr,
        quarter_autocorr=quarter_autocorr,
        dow=dow,
    )
    plain = dict(
        features_days=len(features_days),
        data_days=len(df.columns),
        n_ases=len(df),
        data_start=data_start,
        data_end=data_end,
        features_end=features_end

    )

    print args.data_dir
    #print tensors
    # Store data to the disk
    VarFeeder(args.data_dir, tensors, plain)


if __name__ == '__main__':
    run()
