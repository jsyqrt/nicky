# coding: utf-8
# Long Short Term Memory, Recurrent Neural Network to classify log return.
import os

import numpy as np
import sklearn
import datetime

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

from nicky.data.sohu import sohu
from nicky.data.sae import StackedAutoencoder


np.random.seed(0)
torch.manual_seed(0)


class LSTMC(torch.nn.Module):
    def __init__(self, nb_features=1, hidden_size=100, nb_layers=5, dropout=0.5, time_step=5, lr=0.001, n_class=4):
        super(LSTMC, self).__init__()
        self.nb_features=nb_features
        self.hidden_size=hidden_size
        self.n_class = n_class
        self.nb_layers=nb_layers
        self.lstm = torch.nn.LSTM(
                self.nb_features, self.hidden_size, self.nb_layers, dropout=dropout, )
        self.optimizer = optim.Adam(params=self.parameters(), lr=lr)
        self.lin = torch.nn.Linear(self.hidden_size, self.n_class)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loss = 0
        self.loss_ct = 0
        self.time_step=time_step

    def forward(self, input):
        h0 = Variable(torch.zeros(self.nb_layers, input.size()[1], self.hidden_size))
        c0 = Variable(torch.zeros(self.nb_layers, input.size()[1], self.hidden_size))
        output, hn = self.lstm(input, (h0, c0))
        output = F.sigmoid(self.lin(output[-1]))
        #output = self.lin(output[-1])
        return output

    def gen_batch(self, x, y, time_step=5, batch_size=20, ):
        for batch_idx, i in enumerate(range(0, x.shape[0]-time_step-batch_size, batch_size)):
            xx = np.array([x[j:j+time_step, :] for j in range(i, i+batch_size)])
            yy = y[i+time_step:i+time_step+batch_size]
            yield batch_idx, xx, yy

    def predict(self, x, y, batch_size=5, real_close=None):
        self.train_loss = 0
        self.loss_ct = 0

        print(
        '#'*80 +
        '\ntesting..., batch_size={}, test set length:{}'.format(
            *(batch_size, x.shape[0])))

        target = []
        predicted = []

        for batch_idx, xx, yy in self.gen_batch(x, y, self.time_step, batch_size):

            xx = np.stack(xx)
            xx = Variable(torch.FloatTensor(xx), requires_grad=False)
            xx = torch.transpose(xx, 0, 1)
            yy = Variable(torch.FloatTensor(yy), requires_grad=False).long()

            self.eval()

            out = self.forward(xx)
            loss = self.criterion(out, yy)

            self.train_loss += loss.data[0]
            self.loss_ct += 1

            target.extend(list(yy.data.numpy()))
            _, predictedy = torch.max(out.data, 1)
            predicted.extend(list(predictedy))

            self.train()
            self.optimizer.zero_grad()

            out = self.forward(xx)
            loss = self.criterion(out, yy)

            loss.backward()
            self.optimizer.step()


        target = np.array(target).reshape((len(target), ))
        predicted = np.array(predicted).reshape((len(target), ))

        mse_loss = ((target - predicted)**2).mean(axis=0)
        total_loss = self.train_loss / self.loss_ct

        print(str({
            'temp_mse_loss': mse_loss,
            'total_mse_loss': total_loss,
        }))

        result =  {
            'target': target,
            'predicted': predicted,
            'performance': sohu.performance(target, predicted, True, real_close) if real_close is not None else sohu.performance(target, predicted),
        }
        print('{}'.format(str(result['performance'])))

        return result

    def fit(self, x, y, batch_size=60, n_epoch=500, real_close=None):
        print(
        '#'*80 +
        '\ntraining..., batch_size={}, n_epoch={}, train set length:{}'.format(
            *(batch_size, n_epoch, x.shape[0])))

        for epoch_idx in range(n_epoch):

            target = []
            predicted = []

            for batch_idx, xx, yy in self.gen_batch(x, y, self.time_step, batch_size):

                xx = np.stack(xx)
                xx = Variable(torch.FloatTensor(xx), requires_grad=False)
                xx = torch.transpose(xx, 0, 1)
                yy = Variable(torch.FloatTensor(yy), requires_grad=False).long()

                self.train()
                self.optimizer.zero_grad()

                out = self.forward(xx)
                loss = self.criterion(out, yy)

                self.train_loss += loss.data[0]
                self.loss_ct += 1

                loss.backward()
                self.optimizer.step()

                target.extend(list(yy.data.numpy()))
                _, predictedy = torch.max(out.data, 1)
                predicted.extend(list(predictedy))

            target = np.array(target).reshape((len(target), ))
            predicted = np.array(predicted).reshape((len(target), ))

            mse_loss = ((target - predicted)**2).mean(axis=0)
            total_loss = self.train_loss / self.loss_ct

            print(str({
                'epoch_idx': epoch_idx,
                'temp_mse_loss': mse_loss,
                'total_mse_loss': total_loss,
            }))



        result =  {
            'target': target,
            'predicted': predicted,
            'performance': sohu.performance(target, predicted, True, real_close) if real_close is not None else sohu.performance(target, predicted),
        }
        print('{}'.format(str(result['performance'])))
        return result


if __name__ == '__main__':

    data = sohu.get_hist_data('zs_000001', indicators=True, preprocess=True)
    restore = data.restore

    split = int(data.shape[0] * 0.8)
    trainX = data.iloc[:split, :]
    trainY = data.iloc[1:split+1, 5]
    testX = data.iloc[split:-1, :]
    testY = data.iloc[split+1:, 5]

    q = lambda x: (x.quantile(0.25), x.quantile(0.5), x.quantile(0.75), )
    def f(x, y):
        qq = q(y)
        if x < qq[0]:
            return 0
        elif ((x >= qq[0]) & (x < qq[1])):
            return 1
        elif ((x >= qq[1]) & (x < qq[2])):
            return 2
        else:
            return 3

    def f(x, y):
        if x > 0:
            return 1
        else:
            return 0

    trainy = np.array([f(i, trainY) for i in trainY])
    testy = np.array([f(i, testY) for i in testY])

    trainY = data.iloc[1:split+1, 3]
    testY = data.iloc[split+1:, 3]

    ts = lambda x: x.values
    trainX, trainY, testX, testY = ts(trainX), ts(trainY), ts(testX), ts(testY)
    rst = lambda x, i: x * restore['std'][i] + restore['mean'][i]
    trainY = rst(trainY, 'close')
    testY = rst(testY, 'close')

#    sae = StackedAutoencoder([trainX.shape[1], 10, 10, 10, 10], )
#    sae.train(trainX)
#    trainX = sae.transform(trainX)
#    testX = sae.transform(testX)

    batch_size = 50
    n_epoch = 500

    lstmc_path = 'models/lstmc-without-sae-sigmoid-{}-epoch-{}-batch'.format(n_epoch, batch_size)

    lstmc = LSTMC(trainX.shape[1], time_step=5, n_class=2)

    if os.path.exists(lstmc_path):
        lstmc.load_state_dict(torch.load(lstmc_path))
    else:
        train_result = lstmc.fit(trainX, trainy, n_epoch=n_epoch, batch_size=batch_size, real_close=trainY)
        print(train_result)
        torch.save(lstmc.state_dict(), lstmc_path)

    test_result = lstmc.predict(testX, testy, batch_size=1, real_close=testY)

    print(test_result)
