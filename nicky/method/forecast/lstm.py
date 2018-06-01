# coding: utf-8
# Long Short Term Memory, Recurrent Neural Network

# TODO tweak the APIs to make it more useful.

import random
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

from nicky.data.sae import StackedAutoencoder
from nicky.data.sohu import sohu


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

logger = logging.getLogger(__name__)


class LSTM(torch.nn.Module):
    def __init__(self, nb_features=1, hidden_size=100, nb_layers=5,
                        dropout=0.5, time_step=5, lr=0.001):

        super(LSTM, self).__init__()
        self.nb_features=nb_features
        self.hidden_size=hidden_size
        self.nb_layers=nb_layers
        self.lstm = torch.nn.LSTM(
                self.nb_features, self.hidden_size, self.nb_layers, dropout=dropout)
        self.optimizer = optim.Adam(params=self.parameters(), lr=lr)
        self.lin = torch.nn.Linear(self.hidden_size,1)
        self.criterion = torch.nn.MSELoss()
        self.train_loss = 0
        self.loss_ct = 0
        self.time_step=time_step

        logger.warning(
        '#'*80 +
        '\ncreated LSTM instance. with nb_features:{}, \
                hidden_size:{}, nb_layers:{}, dropout:{}, time_step:{}, \
                lr:{}'.format(
            *(nb_features, hidden_size, nb_layers, dropout, time_step, lr), )
        )

    def forward(self, input):
        h0 = Variable(torch.zeros(self.nb_layers, input.size()[1], self.hidden_size))
        c0 = Variable(torch.zeros(self.nb_layers, input.size()[1], self.hidden_size))
        output, hn = self.lstm(input, (h0, c0))
        output = F.relu(self.lin(output[-1]))
        #output = self.lin(output[-1])
        return output

    def gen_batch(self, x, y, time_step=5, batch_size=20, ):
        for batch_idx, i in enumerate(range(0, x.shape[0]-time_step-batch_size, batch_size)):
            xx = np.array([x[j:j+time_step, :] for j in range(i, i+batch_size)])
            yy = y[i+time_step:i+time_step+batch_size]
            yield batch_idx, xx, yy

    def predict(self, x, y, batch_size=5, n_epoch=10, show_all=False):
        self.train_loss = 0
        self.loss_ct = 0

        logger.warning(
        '#'*80 +
        '\ntesting..., batch_size={}, n_epoch={}, test set length:{}'.format(
            *(batch_size, n_epoch, x.shape[0])))

        predicted = self.fit(x, y, batch_size, n_epoch=(1, n_epoch), show_all=show_all)

        return predicted

    def fit(self, x, y, batch_size=60, n_epoch=(50, 10), show_all=False):
        logger.warning(
        '#'*80 +
        '\ntraining..., batch_size={}, n_epoch={}, train set length:{}'.format(
            *(batch_size, n_epoch, x.shape[0])))

        for epoch_idx in range(n_epoch[0]):

            target = []
            predicted = []

            datas = []
            for batch_idx, xx, yy in self.gen_batch(x, y, self.time_step, batch_size):

                xx = np.stack(xx)
                xx = Variable(torch.FloatTensor(xx), requires_grad=False)
                xx = torch.transpose(xx, 0, 1)
                yy = Variable(torch.FloatTensor(yy), requires_grad=False)

                datas.append({'xx':xx, 'yy':yy, 'idx': batch_idx+1})

                self.eval()
                out = self.forward(xx)

                predicted.extend(list(out.data.numpy()))
                target.extend(list(yy.data.numpy()))

                for epoch in range(n_epoch[1]):
                    count = len(datas)
                    while count > 0:
                        for data in datas:
                            xx = data['xx']
                            yy = data['yy']
                            ctxy = data['idx']
                            if ctxy == 0:
                                continue
                            else:
                                data['idx'] = ctxy - 1

                                self.train()
                                self.optimizer.zero_grad()

                                out = self.forward(xx)
                                loss = self.criterion(out, yy)

                                self.train_loss += loss.data[0]
                                self.loss_ct += 1

                                loss.backward()
                                self.optimizer.step()
                        count -= 1

                if show_all:
                    t = np.array(target).reshape((len(target), ))
                    p = np.array(predicted).reshape((len(predicted), ))

                    mse_loss = ((np.array(t) - np.array(p))**2).mean(axis=0)
                    total_loss = self.train_loss / self.loss_ct

                    logger.info(str({
                        'epoch_idx': epoch_idx,
                        'batch_idx': batch_idx,
                        'temp_mse_loss': mse_loss,
                        'total_mse_loss': total_loss,
                    }))

            target = np.array(target).reshape((len(target), ))
            predicted = np.array(predicted).reshape((len(predicted), ))

            mse_loss = ((target - predicted)**2).mean(axis=0)
            total_loss = self.train_loss / self.loss_ct

            logger.info(str({
                'epoch_idx': epoch_idx,
                'temp_mse_loss': mse_loss,
                'total_mse_loss': total_loss,
            }))

        return predicted

if __name__ == '__main__':

    data = sohu.get_hist_data('zs_000001', indicators=True, preprocess=True)
    restore = data.restore

    split = int(data.shape[0] * 0.6)
    trainX = data[:split, :]
    trainY = data[1:split+1, 3]
    testX = data[split:-1, :]
    testY = data[split+1:, 3]

#    sae = StackedAutoencoder([trainX.shape[1], 10, 10, 10, 10], )
#    sae.train(trainX)
#    trainX = sae.transform(trainX)
#    testX = sae.transform(testX)

    lstm = LSTM(trainX.shape[1])

    predtrain = lstm.fit(trainX, trainY,)
    trainY = trainY[:len(predtrain)]
    print(len(trainY), len(predtrain))
    print(sohu.performance(trainY, predtrain))
    print(sohu.performance(
        trainY*restore['std']['close'] + restore['mean']['close'],
        predtrain*restore['std']['close'] + restore['mean']['close'],)
    )

    predtest  = lstm.predict(testX, testY)
    testY = testY[:len(predtest)]
    print(sohu.performance(testY, predtest))
    print(sohu.performance(
        testY*restore['std']['close'] + restore['mean']['close'],
        predtest*restore['std']['close'] + restore['mean']['close'],)
    )

