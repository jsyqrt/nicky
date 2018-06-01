# coding: utf-8
# experiments.py
# 
import os

import numpy as np
import sklearn
import datetime

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import logging

import warnings
warnings.filterwarnings('ignore')

from nicky.data.sohu import sohu
from nicky.data.sae import StackedAutoencoder
from nicky.data.papermethod import LSTM
from nicky.data.classify import LSTMC


np.random.seed(0)
torch.manual_seed(0)

n_epochs = [1000, ]
batch_sizes = [50, ]
pbatch_sizes = [1, ]
time_steps = [5, ]
shuffles = [True, ]
autoencodes = [False, True]
wavelets = [True, ]

def exp_all_with_classify(code, start, end, log_file, logger):
    logger.info('exp_all_with_classify...')
    results = []
    for shuffle in shuffles:
        for wavelet in wavelets:
            logger.info('reading data...code:{}, start={}, end={}, shuffle={}, wavelet={}'.format(code, start, end, shuffle, wavelet))

            datas = sohu.get_prepared_data(code, start, end, shuffle, if_wavelet=wavelet)
            for time_spt, data in datas.items():
                (trainX, trainY, trainZ), (valX, valY, valZ), (testX, testY, testZ), restore = data
                tobinary = lambda x: np.array([1 if i>0 else 0 for i in x])
                trainZ, valZ, testZ = tobinary(trainZ), tobinary(valZ), tobinary(testZ)
                for autoencode in autoencodes:
                    if autoencode:
                        logger.info('encoding data...')
                        sae = StackedAutoencoder([trainX.shape[1], 10, 10, 10, 10])
                        sae.train(trainX)
                        tsf = lambda x: sae.transform(x)
                        ttrainX, tvalX, ttestX = tsf(trainX), tsf(valX), tsf(testX)
                    for n_epoch in n_epochs:
                        for batch_size in batch_sizes:
                            for time_step in time_steps:
                                logger.info('training data...n_epoch:{}, batch_size={}, time_step={}'.format(n_epoch, batch_size, time_step))
                                rst = lambda x, i: x * restore['std'][i] + restore['mean'][i]

                                lstm = LSTM(ttrainX.shape[1] if autoencode else trainX.shape[1], time_step=time_step)
                                lstmc = LSTMC(trainX.shape[1], time_step=time_step, n_class=2)
                                file_path_base = 'code-{}-start-{}-end-{}-shuffle-{}-wavelet-{}-time_spt-{}-autoencode-{}-n_epoch-{}-batch_size-{}-time_step-{}'.format(
                                    code,
                                    start,
                                    end,

                                    shuffle,
                                    wavelet,
                                    '-'.join(time_spt),

                                    autoencode,

                                    n_epoch,
                                    batch_size,

                                    time_step,
                                )

                                lstm_path = 'models/lstm-{}'.format(file_path_base)
                                lstmc_path = 'models/lstmc-{}'.format(file_path_base)

                                if os.path.exists(lstm_path):
                                    lstm.load_state_dict(torch.load(lstm_path))
                                else:
                                    lstm_train_result = lstm.fit(ttrainX if autoencode else trainX, trainY, n_epoch=n_epoch, batch_size=batch_size, restore=restore)

                                    torch.save(lstm.state_dict(), lstm_path)

                                trainY = rst(trainY, 'close')

                                if os.path.exists(lstmc_path):
                                    lstmc.load_state_dict(torch.load(lstmc_path))
                                else:
                                    lstmc_train_result = lstmc.fit(trainX, trainZ, n_epoch=n_epoch, batch_size=batch_size, real_close=trainY)

                                    torch.save(lstmc.state_dict(), lstmc_path)

                                for pbatch_size in pbatch_sizes:

                                    lstm_val_result = lstm.predict(tvalX if autoencode else valX, valY, batch_size=pbatch_size,restore=restore)
                                    valY = rst(valY, 'close')
                                    lstmc_val_result = lstmc.predict(valX, valZ, batch_size=pbatch_size,real_close=valY)
                                    combine_val_result = sohu.performance(lstm_val_result['target'], lstm_val_result['predicted'], True, lstmc_val_result['predicted'], True)

                                    lstm_test_result = lstm.predict(ttestX if autoencode else testX, testY, batch_size=pbatch_size, restore=restore)
                                    testY = rst(testY, 'close')
                                    lstmc_test_result = lstmc.predict(testX, testZ, batch_size=pbatch_size, real_close=testY)
                                    combine_test_result = sohu.performance(lstm_test_result['target'], lstm_test_result['predicted'], True, lstmc_test_result['predicted'], True)

                                result = ' | '.join([
                                    code,
                                    start,
                                    end,
                                    str(shuffle),
                                    str(wavelet),
                                    '-'.join(time_spt),

                                    str(autoencode),
                                    str(n_epoch),
                                    str(batch_size),
                                    str(time_step),
                                    str(pbatch_size),

                                    str(lstm_val_result['performance']),
                                    str(lstmc_val_result['performance']),
                                    str(combine_val_result),

                                    str(lstm_test_result['performance']),
                                    str(lstmc_test_result['performance']),
                                    str(combine_test_result),
                                ])
                                logger.info('\nresult:\n' + result + '\n\n')
                                with open(log_file, 'a') as f:
                                    f.write(str(result) + '\n')
                                print('#' * 80)
                                print('\n'.join([
                                    str(lstm_val_result['performance']),
                                    str(lstmc_val_result['performance']),
                                    str(combine_val_result),

                                    str(lstm_test_result['performance']),
                                    str(lstmc_test_result['performance']),
                                    str(combine_test_result),
                                    ]))
                                print('#' * 80)

if __name__ == '__main__':
    code = 'zs_000001'
    trange = ('2015-01-01', '2018-01-01')

    logger = logging.getLogger('lab1')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('log/lab-papermethod-with-classify{}.log'.format(str(datetime.datetime.now())))
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh.setFormatter(formatter)

    logger.addHandler(fh)

    exp_all_with_classify(code, trange[0], trange[1], log_file='log/lab-papermethod-with-classify.result', logger=logger)
