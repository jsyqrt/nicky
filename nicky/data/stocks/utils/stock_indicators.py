import warnings

import numpy as np
import pandas as pd

def get_data_indicators(data):
    """Get financial indicators of data.

    Available Indicators Include:
        MACD, CCI, ATR, MA, STD, LBAND, UBAND, EMA20, MA5, MA10, ROC,
            STOK, STOD, WVAD, MTM6, MTM12.

    Args:
        data (pd.DataFrame): The DataFrame to be processed.

    Returns:
        pd.DataFrame: pd.df with indicators columns added.

    """
    target = 'close'

    if target not in data.columns:
        raise KeyError('{} not in pd.columns'.format(target))

    if data.shape[0] < 240:
        warnings.warn('data.shape: {}, less than 240.'.format(data.shape[0]))
        return data

    # MACD Moving average convergence divergence: displays trend following
    # characteristics and momentum characteristics.
    # https://www.linkedin.com/pulse/python-tutorial-macd-moving-average-andrew-hamlet

    data['MACD'] = pd.ewma(data[target], span=26) - pd.ewma(data[target], span=12)

    # CCI Commodity channel index: helps to find the start and the end of a trend.
    # https://www.quantinsti.com/blog/build-technical-indicators-in-python/

    ndays = 20
    TP = (data['high'] + data['low'] + data[target]) / 3
    data['CCI'] = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)))

    # ATR Average true range: measures the volatility of price.
    # https://en.wikipedia.org/wiki/Average_true_range

    TR = lambda high, low, close_prev: max([high-low, abs(high-close_prev), abs(low-close_prev)])

    n = len(data.index)
    data['close_prev'] = data[target].shift()
    data['close_prev'][0] = data[target][0]
    TRs = [TR(i['high'], i['low'], i['close_prev']) for j, i in data.iterrows()]
    ATR0 = sum(TRs) / n
    ATRs = [ATR0]
    for i in range(1, n):
        ATRi = (ATRs[i-1] * (n-1) + TRs[i]) / n
        ATRs.append(ATRi)
    data['ATR'] = pd.Series(ATRs, index=data.index)
    del data['close_prev']

    # BOLL Bollinger Band: provides a relative definition of high and low,
    # which aids in rigorous
    # https://medium.com/python-data/setting-up-a-bollinger-band-with-python-28941e2fa300

    data['MA'] = data[target].rolling(window=ndays).mean()
    data['STD'] = data[target].rolling(window=ndays).std()
    data['LBAND'] = data['MA'] + (data['STD'] * 2)
    data['UBAND'] = data['MA'] - (data['STD'] * 2)

    # EMA20 20 day Exponential Moving Average
    # https://www.quantinsti.com/blog/build-technical-indicators-in-python/

    data['EWMA'] = pd.ewma(data[target], span=ndays)

    # MA5/MA10 5/10 day Moving Average

    data['MA5'] = pd.rolling_mean(data[target], window=5)
    data['MA10'] = pd.rolling_mean(data[target], window=10)

    # Price rate of change: shows the speed at which a stock’s price is changing
    # https://www.quantinsti.com/blog/build-technical-indicators-in-python/

    data['ROC'] = data[target].diff(ndays) / data[target].shift(ndays)

    # Stochastic Momentum Index: shows where the close price is relative to the midpoint of the same range.
    # http://www.andrewshamlet.net/2017/07/13/python-tutorial-stochastic-oscillator/

    data['STOK'] = ((data[target] - pd.rolling_min(data['low'], ndays)) / (pd.rolling_max(data['high'], ndays) - pd.rolling_min(data['low'], ndays))) * 100
    data['STOD'] = pd.rolling_mean(data['STOK'], 3)

    # WVAD Williams’s Variable Accumulation/Distribution: measures the buying and selling pressure.
    # http://www.ensignsoftware.com/blog/2012/01/williams-variable-accumulation-distribution/
    # WVAD = (( Close – Open ) / ( High – Low )) * Volume

    data['WVAD'] = pd.Series([((j[target]-j['open'])/(j['high']-j['low']))*j['volume'] for i, j in data.iterrows()], index=data.index)

    # MTM6/MTM12 6/12 month Momentum: helps pinpoint the end of a decline or advance
    # https://github.com/kylejusticemagnuson/pyti/blob/master/pyti/momentum.py

    ndays = 120
    data['MTM6'] = [np.nan] * (ndays-1) + list([data.iloc[i][target] - data.iloc[i+1-ndays][target] for i in range(ndays-1, len(data.index))])
    ndays = 240
    data['MTM12'] = [np.nan] * (ndays-1) + list([data.iloc[i][target] - data.iloc[i+1-ndays][target] for i in range(ndays-1, len(data.index))])

    return data

