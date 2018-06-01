# coding: utf-8
# crycompare.py
import os
import datetime

import numpy as np
import pandas as pd

from utils import *

__dir__ = os.path.split(os.path.realpath(__file__))[0]

class crycompare:
    __data_dir = os.path.join(__dir__, 'data', 'crycompare')
    __data_daily_dict = os.path.join(__data_dir, 'crycompare.dict')
    __data_currency_basics = os.path.join(__data_dir, 'crycompare.basics')

    @staticmethod
    def get_hist_data(code, start='2015-01-01', end='2018-01-01', ktype='D'):
        crypto_dict = load_obj(crycompare.__data_daily_dict)
        data = crypto_dict.get(code, None)
        data = data[data.apply(
            lambda x: \
            (x.name <= datetime.datetime.strptime(end, '%Y-%m-%d')) and \
            (x.name >= datetime.datetime.strptime(start, '%Y-%m-%d')), axis=1)]

        return data

    @staticmethod
    def get_currency_basics():
        currency_basics = load_obj(crycompare.__data_currency_basics)
        data = pd.DataFrame(columns=[
                'Algorithm', 'CoinName', 'FullName', 'FullyPremined', 'Id',
                'ImageUrl', 'Name', 'PreMinedValue', 'ProofType', 'SortOrder',
                'Sponsored', 'Symbol', 'TotalCoinSupply', 'TotalCoinsFreeFloat', 'Url'])
        for d in currency_basics['Data']:
            try:
                data.loc[d] = currency_basics['Data'][d]
            except:
                pass
        data.BaseImageUrl = 'https://www.cryptocompare.com'
        data.BaseLinkUrl = 'https://www.cryptocompare.com'

        return data
