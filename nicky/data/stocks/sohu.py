# coding: utf-8
# sohu.py
import os
import math
import json
import urllib
import datetime
import dateutil
import sqlite3
from functools import reduce
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pyflux as pf

import pywt
import tushare as ts

from nicky.utils import config, join_with_rc
from nicky.data.stocks.utils import stock_indicators

__dir__ = join_with_rc(config.db.db_dir)


class sohu:

    __data_src = 'local_db'
    __data_dir = os.path.join(__dir__, 'stocks', 'sohu')
    __ALL_TIME = False

    @staticmethod
    def get_data_from_db(code, start, end, tablename, db_conn, describe=True, indicators=False):
        data = pd.read_sql_query(
            'select * from %s where %s.code="%s" and \
                %s.datetime>="%s" and %s.datetime<="%s";'\
             % (tablename, tablename, code, tablename, start, tablename, end),
            con=db_conn)
        db_conn.close()
        data.index = data['datetime']
        del data['code']
        del data['datetime']
        data = sohu.get_data_describe(data, describe)
        data = sohu.get_data_indicators(data, indicators)
        return data

    @staticmethod
    def weighted_mean(codes, start='2015-01-01', end='2018-01-01', weight_standard='average'):
        """Weighted mean of a group of stocks indexed by `codes`.

        Args:
            codes: list of code.
            start: string of start date, format "%Y-%m-%d".
            end: string like `start`.
            weight_standard: string, "average" for average mean,
                "optimal_max_return" for a optimal portfolio with max return,
                "optimal_min_risk" for a optimal portfolio with min risk.
        Returns:
            pandas.DataFrame. the weight mean of the given codes.
        """

        datas = [sohu.get_hist_data(code, start=start, end=end, describe=True, indicators=True, preprocess=True) for code in codes]

        if weight_standard == 'average':
            restore = {
                'mean': sum([data.restore['mean'] for data in datas]) / len(datas),
                'std': sum([data.restore['std'] for data in datas]) / len(datas),
            }

            datas = pd.concat(datas)
            datas = datas.groupby(datas.index)
            data  = datas.mean()
            data.restore = restore
            return data

        elif weight_standard.startswith('optimal'):
            portfolio = sohu.portfolio(codes, start, end)
            if weight_standard == 'optimal_max_return':
                weights = portfolio['max_sharp_weights']

            elif weight_standard == 'optimal_min_risk':
                weights = portfolio['min_variance_weights']

            shortest = [None, 0]
            for i, code in enumerate(codes):
                if shortest[1] == 0 or datas[i].shape[0] < shortest[1]:
                    shortest[0] = i
                    shortest[1] = datas[i].shape[0]

            shortest_index = datas[shortest[0]].index
            for i, data in enumerate(datas):
                restore = datas[i].restore
                datas[i] = datas[i].loc[shortest_index]
                datas[i].restore = restore

            restore = {
                'mean': sum([datas[i].restore['mean'] * weights[i] for i in range(len(weights))]) / len(weights),
                'std': sum([datas[i].restore['std'] * weights[i] for i in range(len(weights))]) / len(weights),
            }
            data = reduce(lambda x, y: x+y, [datas[i] * weights[i] for i in range(len(weights))])
            data.restore = restore
            return data


    @staticmethod
    def get_data_indicators(data, indicators=False):
        if (indicators is False):
            return data
        data = stock_indicators.get_data_indicators(data)
        return data

    @staticmethod
    def get_hist_data(code, start='2015-01-01', end='2018-01-01', ktype='d', describe=True, indicators=False, preprocess=False, if_wavelet=True):
        if sohu.__data_src == 'local_db':
            conn = sqlite3.connect(os.path.join(sohu.__data_dir, 'sohu.db'))
            dtype = code[:2]
            code = code[3:]
            if dtype == 'cn':
                table_name = 'stock'
                stock_basics = sohu.get_stock_basics()
                infos = stock_basics.loc[code]
            elif dtype == 'zs':
                table_name = 'zsindex'
                index_basics = sohu.get_index_basics()
                infos = index_basics.loc[code]
            else:
                return None
            data = sohu.get_data_from_db(code, start, end, table_name, conn, describe=describe, indicators=indicators)

            restore = {'mean': data.min(), 'std': (data.max()-data.min())}
            #restore = {'mean': data.mean(), 'std': data.std()}

            if len(data.index) == 0:
                data = None
            if preprocess and data is not None:
                data = sohu.preprocess(data, if_wavelet)
            data.infos = infos
            data.restore = restore
            return data
        elif sohu.__data_src == 'remote':
            return None

    @staticmethod
    def get_prepared_data(code, start, end, if_shuffle, if_wavelet=True):
        data = sohu.get_hist_data(code, start, end, indicators=True, preprocess=True, if_wavelet=if_wavelet)
        splits, _ = sohu.time_splits(start, end)
        datas = {}
        datec = lambda datestr: datetime.datetime.strptime(datestr, '%Y-%m-%d')
        for spt in splits:
            train = data[(data.index >= datec(spt[0])) & (data.index <= datec(spt[1])) ]
            val = data[(data.index >= datec(spt[1])) & (data.index <= datec(spt[2])) ]
            test = data[(data.index >= datec(spt[2])) & (data.index <= datec(spt[3])) ]
            if if_shuffle:
                train = train.sample(frac=1)
            tonp = lambda x: (x.values[:-1, :], x.values[1:, 3], x.values[1:, 5])
            datas[spt] = (tonp(train), tonp(val), tonp(test), data.restore)

        return datas

    @staticmethod
    def time_splits(start='2015-01-01', end='2018-01-01'):
        start = datetime.datetime.strptime(start, '%Y-%m-%d').date()
        end = datetime.datetime.strptime(end, '%Y-%m-%d').date()

        tms = []
        while start <= end:
            tms.append(start.strftime('%Y-%m-%d'))
            start += dateutil.relativedelta.relativedelta(months=3)

        splits = list(zip(tms[: -10], tms[8: -2], tms[9: -1], tms[10: ]))
        return splits, len(splits)


    @staticmethod
    def performance(target, predicted, restore=None, restore_column='close', plot=False):
        length = min(len(target), len(predicted))
        target = target[:length]
        predicted = predicted[:length]
        if restore:
            target = target * restore['std'][restore_column] + restore['mean'][restore_column]
            predicted = predicted * restore['std'][restore_column] + restore['mean'][restore_column]

            if restore_column in ['simple return', 'log return', 'volatility']:
                target = [1 if i > 0 else 0 for i in target]
                predicted = [1 if i > 0 else 0 for i in predicted]
                return sum([1-(target[i] - predicted[i])**2 for i in range(length)])/length


        if plot:
            import matplotlib.pyplot as plt

            plt.plot(target)
            plt.plot(predicted)
            plt.legend(('target', 'predicted'))
            plt.show()

        return sohu._performance(target, predicted)

    @staticmethod
    def _performance(target, predicted):

        target = target
        predicted = predicted

        B = 0.0005
        S = 0.0005

        n = target.shape[0]
        rcb = target - target.mean()
        pcb = predicted - predicted.mean()

        MAPE = (1 / n) * sum([abs(rc / (target[i] if target[i] != 0 else 0.000001)) for i, rc in enumerate(target - predicted)])
        R = (rcb * pcb).sum() / ((np.sqrt((rcb**2).sum()) * np.sqrt((pcb**2).sum())) if (np.sqrt((rcb**2).sum()) * np.sqrt((pcb**2).sum())) != 0 else 0.000001)
        TheilU = np.sqrt(((target - predicted)**2).sum() / n) / (np.sqrt((target**2).sum() / n) + np.sqrt((predicted**2).sum() / n))

        deals = [1 if target[i] < predicted[i+1] else 0 for i in range(n - 1)]
        target_deals = [1 if target[i] < target[i+1] else 0 for i in range(n - 1)]
        DA = sum([1 if target_deals[i] == deals[i] else 0 for i in range(n - 1)]) / (n - 1) * 100

#        Returns = ((reduce(lambda x, y: x * y,
#                        [{  1: ((target[i+1] - target[i] - (target[i] * B + target[i+1] * S)) / target[i]),
#                            0: ((target[i] - target[i+1] - (target[i+1] * B + target[i] * S)) / target[i]),
#                            -1: 0,
#                        }[deal] + 1
#                            for i, deal in enumerate(deals)]) ** (1 / (n - 1))) ** (250) - 1) * 100

        Returns = reduce(lambda x, y: x + y,
                        [{  1: ((target[i+1] - target[i] - (target[i] * B + target[i+1] * S)) / target[i]),
                            0: ((target[i] - target[i+1] - (target[i+1] * B + target[i] * S)) / target[i]),
                            -1: 0,
                        }[deal]
                            for i, deal in enumerate(deals)]) * 100 /n * 250

        return {
            'MAPE': MAPE,
            'R': R,
            'TheilU': TheilU,
            'Returns': Returns,
            'DA': DA,
        }


    @staticmethod
    def preprocess(df, if_wavelet=True):
        df = sohu.wavelet(df) if if_wavelet else df
        df = sohu.standardize(df)
        df = df.fillna(df.mean().mean())
        return df

    @staticmethod
    def wavelet(df):
        data = df.values
        data = pywt.wavedec(data, 'haar', level=2, axis=0)
        data[2][:, :] = 0
        data = pywt.waverec(data, 'haar', axis=0)
        df.iloc[:, :] = data[:len(df.index), ]
        return df

    @staticmethod
    def standardize(df):
        logreturn = df['log return']
        df = (df - df.min())/(df.max() - df.min())
        df['log return'] = logreturn
#df = (df - df.mean())/(df.std())
        return df

    @staticmethod
    def get_data_describe(data, describe=True):
        data.index = pd.to_datetime(data.index)
        if sohu.__ALL_TIME:
            x_index = pd.date_range(start=data.index[0], end=data.index[-1], freq='D')
            data = data.reindex(x_index)
        if describe:
            data['log return'] = pd.Series([0] + list(np.diff(np.log(data['close']))), index=data.index)
            data['simple return'] = data['log return'].apply(lambda x: math.exp(x) - 1)
            data['volatility'] = pd.Series([0] + list(np.diff(data['close'])), index=data.index)
            data.Skewness = data.skew()
            data.Kurtosis = data.kurt()
        return data

    #'''
    @staticmethod
    def get_data_forecasting(data, target='close', method='ARIMA', forward_step=10, fit_method='MLE', **kw):
        #method: {'ARIMA', 'DAR', 'CNN', 'WaveNet'}
        if method == 'ARIMA':

            model = pf.ARIMA(data=data, target=target, **kw)
            fit_info = model.fit(fit_method)
            predict = model.predict(forward_step)
            return [predict, fit_info]
        elif method == 'DAR':
            model = pf.DAR(data=data, target=target, **kw)
            fit_info = model.fit(fit_method)
            predict = model.predict(forward_step)
            return [predict, fit_info]
        else:
            pass
    #'''

    @staticmethod
    def get_dist_matrix(zs_or_cn, start='2017-01-01', end='2018-01-01'):
        conn = sqlite3.connect(os.path.join(sohu.__data_dir, 'sohu.db'))
        tablename = '_'.join(
            [
                zs_or_cn,
                start.replace('-', '_'),
                end.replace('-', '_'),
                '200',
                'close',
                'distM',
            ]
        )

        codes = pd.read_sql_query('select * from {};'.format(tablename+'_codes'), con=conn)['code']
        values = pd.read_sql_query('select * from {};'.format(tablename+'_values'), con=conn)['values']

        import scipy.spatial.distance as distC
        #distMscp = distC.pdist
        squareform = distC.squareform

        codes = list(codes)
        values = squareform(values.values)
        data = pd.DataFrame(values, columns=codes, index=codes)
        data.index.name = 'code'

        return data

    @staticmethod
    def get_closest_list(code):
        if code[:2] not in ['zs', 'cn']:
            return None
        data = sohu.get_dist_matrix(code[:2])
        if code in data.index:
            closest = data.loc[code]
            all_code = list(data.columns)
            closest_codes = [all_code[i] for i in closest.values.argsort()]
            return closest_codes
        else:
            return None

    @staticmethod
    def get_stock_basics():
        if sohu.__data_src == 'local_db':
            conn = sqlite3.connect(os.path.join(sohu.__data_dir, 'sohu.db'))
            data = pd.read_sql_query('select * from stocks;', con=conn)
            conn.close()
            data.index = data['code']
            del data['code']
            return data
        elif sohu.__data_src == 'remote':
            return None

    @staticmethod
    def get_index_basics():
        if sohu.__data_src == 'local_db':
            conn = sqlite3.connect(os.path.join(sohu.__data_dir, 'sohu.db'))
            data = pd.read_sql_query('select * from zsindexes;', con=conn)
            conn.close()
            data.index = data['code']
            del data['code']
            return data
        elif sohu.__data_src == 'remote':
            return None

    @staticmethod
    def get_area_basics():
        area_stock_dict = sohu.get_stock_basics().groupby(['area']).groups
        return area_stock_dict

    @staticmethod
    def get_area_mean(area_name='上海', start='2015-01-01', end='2018-01-01', db_name='sohu.db', from_db=True, describe=True):
        if from_db:
            db_name = os.path.join(sohu.__data_dir, db_name)
            conn = sqlite3.connect(db_name)
            result = pd.read_sql_query('select * from area where area_name="%s" and datetime>="%s" and datetime<="%s"' % (area_name, start, end), con=conn)
            result.index = result['datetime']
            del result['datetime']
            del result['area_name']
            result.codes = sohu.get_area_basics()[area_name]
            result.name = area_name
            result.num = len(result.codes)
            result = sohu.get_data_describe(result, describe)
            print('area from db')

        else:
            d = sohu.get_area_basics()
            codes = d[area_name]
            num = len(codes)
            datas = [sohu.get_hist_data('cn_%s' % code, start=start, end=end, describe=False) for code in codes]
            result = pd.concat(datas).groupby('datetime').mean()
            result.codes = codes
            result.name = area_name
            result.num = num
            result = sohu.get_data_describe(result, describe)
            print('area with cal')
        return result


    @staticmethod
    def get_concept_basics(from_db=True, db_name='sohu.db'):
        if not from_db:
            concept_stock_dict = ts.get_concept_classified()
            concept_stock_dict.index = concept_stock_dict['code']
            codes = list(sohu.get_stock_basics().index)
            x = concept_stock_dict.index
            for index in x:
                if index not in codes:
                    concept_stock_dict =concept_stock_dict.drop([index])
            concept_stock_dict = concept_stock_dict.groupby(['c_name']).groups
            print('concepts dict from remote')
            return concept_stock_dict
        else:
            db_name = os.path.join(sohu._sohu__data_dir, db_name)
            conn = sqlite3.connect(db_name)
            result = pd.read_sql_query('select * from concepts', con=conn)
            result.index = result['code']
            del result['code']
            concept_stock_dict = result.groupby('concept_name').groups
            print('concepts dict from db')
            return concept_stock_dict

    @staticmethod
    def get_concept_mean(concept_name='外资背景', start='2015-01-01', end='2018-01-01', db_name='sohu.db', from_db=True, describe=True):
        if from_db:
            db_name = os.path.join(sohu.__data_dir, db_name)
            conn = sqlite3.connect(db_name)
            result = pd.read_sql_query('select * from concept where concept_name="%s" and datetime>="%s" and datetime<="%s"' % (concept_name, start, end), con=conn)
            result.index = result['datetime']
            del result['datetime']
            del result['concept_name']
            result.codes = sohu.get_concept_basics()[concept_name]
            result.name = concept_name
            result.num = len(result.codes)
            result = sohu.get_data_describe(result, describe)
            print('concept from db')

        else:
            d = sohu.get_concept_basics()
            codes = d[concept_name]
            num = len(codes)
            datas = [sohu.get_hist_data('cn_%s' % code, start=start, end=end, describe=False) for code in codes]
            datas = [ i for i in datas if i is not None]
            if len(datas) != 0:
                result = pd.concat(datas).groupby('datetime').mean()
                result.codes = codes
                result.name = concept_name
                result.num = num
                result = sohu.get_data_describe(result, describe)
            else:
                result = None
            print('concept with cal')
        return result

    #'''
    @staticmethod
    def get_industry_basics():
        stocks = sohu.get_stock_basics()
        industry_stock_dict = stocks.groupby(['industry']).groups
        return industry_stock_dict

    @staticmethod
    def get_industry_mean(industry_name='区域地产', start='2015-01-01', end='2018-01-01', db_name='sohu.db', from_db=True, describe=True):
        if from_db:
            db_name = os.path.join(sohu.__data_dir, db_name)
            conn = sqlite3.connect(db_name)
            result = pd.read_sql_query('select * from industry where industry_name="%s" and datetime>="%s" and datetime<="%s"' % (industry_name, start, end), con=conn)
            result.index = result['datetime']
            del result['datetime']
            del result['industry_name']
            result.codes = sohu.get_industry_basics()[industry_name]
            result.name = industry_name
            result.num = len(result.codes)
            result = sohu.get_data_describe(result, describe)
            print('industry from db')

        else:
            d = sohu.get_industry_basics()
            codes = d[industry_name]
            num = len(codes)
            datas = [sohu.get_hist_data('cn_%s' % code, start=start, end=end, describe=False) for code in codes]
            result = pd.concat(datas).groupby('datetime').mean()
            result.codes = codes
            result.name = industry_name
            result.num = num
            result = sohu.get_data_describe(result, describe)
            print('industry with cal')
        return result

    @staticmethod
    def save_industries_to_db(db_name='sohu.db'):
        db_name = os.path.join(sohu._sohu__data_dir, db_name)
        conn = sqlite3.connect(db_name)
        industry_stock_dict = sohu.get_industry_basics()
        for industry_name in industry_stock_dict:
        #if True:
            #industry_name = '区域地产'
            industry_mean = sohu.get_industry_mean(industry_name, '1970-01-01', '2018-01-01')
            industry_mean['industry_name'] = [industry_name for i in range(len(industry_mean.index))]
            industry_mean.to_sql('industry', con=conn, if_exists='append')
            print('done with %s!' % industry_name)
        print('done with industry!')

    @staticmethod
    def save_concepts_to_db(db_name='sohu.db'):
        db_name = os.path.join(sohu._sohu__data_dir, db_name)
        conn = sqlite3.connect(db_name)
        concept_stock_dict = sohu.get_concept_basics()
        for concept_name in concept_stock_dict:
            concept_mean = sohu.get_concept_mean(concept_name, '1970-01-01', '2018-01-01')
            concept_mean['concept_name'] = [concept_name for i in range(len(concept_mean.index))]
            concept_mean.to_sql('concept', con=conn, if_exists='append')
            print('done with %s!' % concept_name)
        print('done with concept!')

    @staticmethod
    def save_concept_stock_dict_to_db(db_name='sohu.db'):
        db_name = os.path.join(sohu._sohu__data_dir, db_name)
        conn = sqlite3.connect(db_name)
        concept_stock_dict = sohu.get_concept_basics(from_db=False)
        result = pd.DataFrame(columns=['concept_name'])
        for concept_name in concept_stock_dict:
            for code in concept_stock_dict[concept_name]:
                result.loc[code] = concept_name
        result.index.name = 'code'
        result.to_sql('concepts', con=conn, if_exists='replace')
        print('done with concept_stock_dict!')


    @staticmethod
    def save_areas_to_db(db_name='sohu.db'):
        db_name = os.path.join(sohu._sohu__data_dir, db_name)
        conn = sqlite3.connect(db_name)
        area_stock_dict = sohu.get_area_basics()
        for area_name in area_stock_dict:
            area_mean = sohu.get_area_mean(area_name, '1970-01-01', '2018-01-01')
            area_mean['area_name'] = [area_name for i in range(len(area_mean.index))]
            area_mean.to_sql('area', con=conn, if_exists='append')
            print('done with %s!' % area_name)
        print('done with area!')


    #'''



    @staticmethod
    def make_db(dbname='sohu.db'):
        dbname = os.path.join(sohu.__data_dir, dbname)
        conn = sqlite3.connect(dbname)
        stocks = sohu.get_stock_basics()
        indexes = sohu.get_index_basics()

        # stock
        for i in range(len(stocks.index)):
            code = stocks.index[i]
            data = sohu.get_hist_data('cn_%s' % code, start='1970-01-01', end='2018-01-01')
            data['code'] = [code for j in range(len(data.index))]
            data.to_sql('stock', con=conn, if_exists='append')
            print('done with stock %s, %f%%' % (code, i/len(stocks.index)*100))
        print('done with stock!')

        # index
        for code in indexes.index:
            data = sohu.get_hist_data('zs_%s' % code, start='1970-01-01', end='2018-01-01')
            data['code'] = [code for i in range(len(data.index))]
            data.to_sql('zsindex', con=conn, if_exists='append')
            print('done with index %s' % code)
        print('done with index!')

        # stocks
        stocks.to_sql('stocks', con=conn)
        print('done with stocks!')

        # indexes
        indexes.to_sql('zsindexes', con=conn)
        print('done with indexes!')

        c = conn.cursor()
        c.execute('create index stock_code on stock(code);')
        c.execute('create index stock_datetime on stock(datetime);')
        c.execute('create index stock_code_datetime on stock(code, datetime);')

        conn.commit()
        conn.close()


    @staticmethod
    def update_db(queue, dbname='sohu.db'):
        dbname = os.path.join(sohu.__data_dir, dbname)
        conn = sqlite3.connect(dbname)

        #'''

        try:

            sohu.update_stock_list()

            l = len(sohu.get_stock_basics().index)
            for i, code in enumerate(sohu.get_stock_basics().index):
                try:
                    sohu.update_stock('cn_%s' % code, conn)

                    queue.put(i/l*100)

                    print('done with stock: %s, %f%%' % (code, i/l*100))
                except:
                    print('failed with stock: %s, %f%%' % (code, i/l*100))

            for code in sohu.get_index_basics().index:
                try:
                    sohu.update_stock('zs_%s' % code, conn)

                    queue.put(i/l *100)

                    print('done with index: %s' % code)
                except:
                    print('failed with index: %s' % code)

            for name in sohu.get_industry_basics().keys():
                try:
                    sohu.update_stock_classes(name, 'industry', conn)
                    print('done with industry: %s' % name)
                except:
                    print('failed with industry: %s' % name)

            for name in sohu.get_concept_basics().keys():
                try:
                    sohu.update_stock_classes(name, 'concept', conn)
                    print('done with concept: %s' % name)
                except:
                    print('failed with concept: %s' % name)

            for name in sohu.get_area_basics().keys():
                try:
                    sohu.update_stock_classes(name, 'area', conn)
                    print('done with area: %s' % name)
                except:
                    print('failed with area: %s' % name)

            print('updated all! commiting...')
            conn.commit()
            print('commited!')

        except Exception as x:
            print(x)
            print('update failed! rollbacking...')
            conn.rollback()
            print('rollbacked!')

        conn.close()


    @staticmethod
    def update_stock(code, conn):
        today = datetime.date.today().strftime('%Y-%m-%d')
        last_date = (sohu.get_hist_data(code, end=today).iloc[-1].name + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

        data = sohu.get_original_data(code, start=last_date, end=today)
        if data is not None:
            data['code'] = [code[-6:] for i in range(len(data.index))]
            data.index.name = 'datetime'
            data.to_sql('stock' if code.startswith('cn') else 'zsindex', con=conn, if_exists='append')

    @staticmethod
    def get_original_data(code='zs_000001', start='2015-01-01', end='2018-01-01'):
        # example: http://q.stock.sohu.com/hisHq?code=cn_600848&start=20170504&end=20180101&order=A&period=d
        # ktype: m month, w week, d day.
        ktype = 'd'
        order = 'A'
        start = start.replace('-', '')
        end = end.replace('-', '')

        url = 'http://q.stock.sohu.com/hisHq?code=%s&start=%s&end=%s&order=%s&period=%s' %(code, start, end, order, ktype)
        try:
            data = urllib.request.urlopen(url).read()
            data = data.decode('utf8')
            data = json.loads(data)
            data = data[0]
            c = data.get('code', '')
            if c != code:
                raise ValueError('Get Wrong Stock with Code: %s' % c)
            h = data.get('hq', [])
            d = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            for i in h:
                open = float(i[1])
                close = float(i[2])
                low = float(i[5])
                high = float(i[6])
                volume = float(i[7])
                d.loc[i[0]] = [open, high, low, close, volume]
            d.name = code
            d.ktype = ktype
            if len(d.index) != 0:
                return d
            else:
                return None
        except:
            #raise ValueError('Error with Getting Stock Hist Data, Code:%s' %code)
            return None

    @staticmethod
    def update_stock_list():
        pass

    @staticmethod
    def update_stock_classes(name, class_name, conn):
        today = datetime.date.today().strftime('%Y-%m-%d')
        if class_name == 'industry':
            last_date = (sohu.get_industry_mean(name, end=today).iloc[-1].name + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            data = sohu.get_industry_mean(name, start=last_date, end=today, from_db=False, describe=False)
            data['industry_name'] = [name for i in range(len(data.index))]
            data.index.name = 'datetime'
            data.to_sql('industry', con=conn, if_exists='append')

        elif class_name == 'concept':
            last_date = (sohu.get_concept_mean(name, end=today).iloc[-1].name + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            data = sohu.get_concept_mean(name, start=last_date, end=today, from_db=False, describe=False)
            if data is not None:
                data['concept_name'] = [name for i in range(len(data.index))]
                data.index.name = 'datetime'
                data.to_sql('concept', con=conn, if_exists='append')

        elif class_name == 'area':
            last_date = (sohu.get_area_mean(name, end=today).iloc[-1].name + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            data = sohu.get_area_mean(name, start=last_date, end=today, from_db=False, describe=False)
            data['area_name'] = [name for i in range(len(data.index))]
            data.index.name = 'datetime'
            data.to_sql('area', con=conn, if_exists='append')

        else:
            pass

    @staticmethod
    def portfolio(codes, start='2015-01-01', end='2018-01-01'):
        '''
        Get portfolio weights of the best return with determined risk level and the lowest risk level with determined wanted return.
        '''
        n = len(codes)

        returns = pd.DataFrame()
        rtns = {}
        shortest = [None, 0]
        for code in codes:
            rtn = sohu.get_hist_data(code, start, end, describe=True)['log return']
            rtns[code] = rtn
            if shortest[1] == 0 or rtn.shape[0] < shortest[1]:
                shortest[0] = code
                shortest[1] = rtn.shape[0]

        shortest_index = rtns[shortest[0]].index
        for code in rtns:
            returns[code] = rtns[code].loc[shortest_index]
        returns.columns = codes

        means = returns.mean() * 252
        covs = returns.cov() * 252

        pret = lambda means, weights: np.sum(means * weights) * 252
        pvol = lambda covs, weights: np.sqrt(np.dot(weights.T, np.dot(covs * 252, weights)))
        sharp = lambda means, covs, weights, rf=0: (pret(means, weights) - rf) / pvol(covs, weights)
        stats = lambda means, covs, weights, rf=0: (pret(means, weights), pvol(covs, weights), sharp(means, covs, weights, rf))

        min_func_sharp = lambda weights: -sharp(means, covs, weights)
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0, 1) for x in range(n))

        import scipy.optimize as sco

        max_sharp_opt = sco.minimize(min_func_sharp, n * [1. / n,], method='SLSQP', bounds=bnds, constraints=cons)

        max_sharp_weights = max_sharp_opt['x'].round(4)
        max_sharp_stats = stats(means, covs, max_sharp_weights)

        min_func_variance = lambda weights: pvol(covs, weights) ** 2
        min_variance_opt = sco.minimize(min_func_variance, n * [1. / n,], method='SLSQP', bounds=bnds, constraints=cons)

        min_variance_weights = min_variance_opt['x'].round(4)
        min_variance_stats = stats(means, covs, min_variance_weights)

        result = {
            'max_sharp_weights': max_sharp_weights,
            'max_sharp_stats': max_sharp_stats,
            'min_variance_weights': min_variance_weights,
            'min_variance_stats': min_variance_stats,
        }

        return result


if __name__ == '__main__':

    x = sohu.get_hist_data('cn_000001')
    y = sohu.performance(x['close'], x['open'])
    z = sohu.performance(x['close'], x['close'])

    print(y)
    print(z)
