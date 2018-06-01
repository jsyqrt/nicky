# coding: utf-8
# prepare_data.py

from sohu import sohu

# train time range = (2006-01-01, 2018-01-01)
# test time range = (2008-01-01, 2018-01-01)

from datetime import datetime
from dateutil.relativedelta import relativedelta

def time_ranges(start='2015-01-01', end='2018-01-01'):
    start = datetime.strptime(start, '%Y-%m-%d').date()
    end = datetime.strptime(end, '%Y-%m-%d').date()

    tms = []
    while start <= end:
        tms.append(start.strftime('%Y-%m-%d'))
        start += relativedelta(months=3)
    
    splits = list(zip(tms[: -10], tms[8: -2], tms[9: -1], tms[10: ]))
    return splits, len(splits)

def get_data(code, time_splits, if_wavelet, if_shuffle):
    t = time_splits
    print(t)
    train = sohu.get_hist_data(code, start=t[0], end=t[1], preprocess=True, if_wavelet=if_wavelet, if_shuffle=if_shuffle)
    val = sohu.get_hist_data(code, start=t[1], end=t[2], preprocess=True, if_wavelet=if_wavelet, )
    test = sohu.get_hist_data(code, start=t[2], end=t[3], preprocess=True, if_wavelet=if_wavelet, )

    tonp = lambda x: (x.values[:-1, :], x.values[1:, 3])
    
    trainX, trainY = tonp(train)
    volX, volY = tonp(vol)
    testX, testY = tonp(test)

    return (trainX, trainY, volX, volY, testX, testY)

def get_all_data(code, time_range, if_wavelet=True, if_shuffle=True):
    time_splits, length = time_ranges(*time_range)
    datas = {}
    for split in time_splits:
        datas[split] = get_data(code, split, if_wavelet, if_shuffle)
    return datas

if __name__ == '__main__':
    code = 'zs_000001'
    trange = ('2015-01-01', '2018-01-01') 
    data = get_all_data(code, trange)

    print(data.keys())
