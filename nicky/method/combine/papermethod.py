# coding: utf-8
# Long Short Term Memory, Recurrent Neural Network
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
from nicky.data.classify import LSTMC


np.random.seed(0)
torch.manual_seed(0)


class LSTM(torch.nn.Module):
    def __init__(self, nb_features=1, hidden_size=100, nb_layers=5, dropout=0.5, time_step=5, lr=0.001):
        super(LSTM, self).__init__()
        self.nb_features=nb_features
        self.hidden_size=hidden_size
        self.nb_layers=nb_layers
        self.lstm = torch.nn.LSTM(
                self.nb_features, self.hidden_size, self.nb_layers, dropout=dropout, )
        self.optimizer = optim.Adam(params=self.parameters(), lr=lr)
        self.lin = torch.nn.Linear(self.hidden_size,1)
        self.criterion = torch.nn.MSELoss()
        self.train_loss = 0
        self.loss_ct = 0
        self.time_step=time_step

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

    def predict(self, x, y, batch_size=5,restore=None):
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
            yy = Variable(torch.FloatTensor(yy), requires_grad=False)

            self.eval()

            out = self.forward(xx)
            loss = self.criterion(out, yy)

            self.train_loss += loss.data[0]
            self.loss_ct += 1

            target.extend(list(yy.data.numpy()))
            predicted.extend(list(out.data.numpy()))

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

        if restore:
            rtarget = target * restore['std']['close'] + restore['mean']['close']
            rpredicted = predicted * restore['std']['close'] + restore['mean']['close']

        result =  {
            'target': target,
            'predicted': predicted,
            'restored_target': rtarget,
            'restored_predicted': rpredicted,
            'performance': sohu.performance(target, predicted),
            'restored_performance': sohu.performance(rtarget, rpredicted),
        }
        print('{}'.format(str(result['performance'])))

        return result

    def fit(self, x, y, batch_size=60, n_epoch=500, restore=None):
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
                yy = Variable(torch.FloatTensor(yy), requires_grad=False)

                self.train()
                self.optimizer.zero_grad()

                out = self.forward(xx)
                loss = self.criterion(out, yy)

                self.train_loss += loss.data[0]
                self.loss_ct += 1

                loss.backward()
                self.optimizer.step()

                target.extend(list(yy.data.numpy()))
                predicted.extend(list(out.data.numpy()))

            target = np.array(target).reshape((len(target), ))
            predicted = np.array(predicted).reshape((len(target), ))

            mse_loss = ((target - predicted)**2).mean(axis=0)
            total_loss = self.train_loss / self.loss_ct

            print(str({
                'epoch_idx': epoch_idx,
                'temp_mse_loss': mse_loss,
                'total_mse_loss': total_loss,
            }))

        if restore:
            rtarget = target * restore['std']['close'] + restore['mean']['close']
            rpredicted = predicted * restore['std']['close'] + restore['mean']['close']

        result =  {
            'target': target,
            'predicted': predicted,
            'restored_target': rtarget,
            'restored_predicted': rpredicted,
            'performance': sohu.performance(target, predicted),
            'restored_performance': sohu.performance(rtarget, rpredicted),
        }
        print('{}'.format(str(result['performance'])))
        return result


if __name__ == '__main__':

    data = sohu.get_hist_data('zs_000001', indicators=True, preprocess=True)
    restore = data.restore
    data= data.values
    split = int(data.shape[0] * 0.8)
    trainX = data[:split, :]
    testX = data[split:-1, :]
    trainY = data[1:split+1, 3]
    testY = data[split+1:, 3]

    tb = lambda x: 1 if x>0 else 0
    trainy = data[1:split+1, 5]
    testy = data[split+1:, 5]
    trainy = np.array([tb(i) for i in trainy])
    testy = np.array([tb(i) for i in testy])

    rst = lambda x, i: x * restore['std'][i] + restore['mean'][i]
    ttrainY = rst(trainY, 'close')
    ttestY = rst(testY, 'close')

    n_epoch = 1000
    batch_size = 50

#    sae = StackedAutoencoder([trainX.shape[1], 10, 10, 10, 10], )
#    sae.train(trainX.copy())
#    trainX = sae.transform(trainX)
#    testX = sae.transform(testX)

    lstm_path = 'models/lstm-without-sae-{}-epoch-{}-batch'.format(n_epoch, batch_size)

    lstm = LSTM(trainX.shape[1], time_step=5)

    if os.path.exists(lstm_path):
        lstm.load_state_dict(torch.load(lstm_path))
    else:
        train_result = lstm.fit(trainX.copy(), trainY.copy(), n_epoch=n_epoch, batch_size=50, restore=restore)
        print(train_result)

        torch.save(lstm.state_dict(), lstm_path)

    test_result = lstm.predict(testX.copy(), testY.copy(), batch_size=1, restore=restore)

    print(test_result)

    lstmc_path = 'models/lstmc-without-sae-sigmoid-{}-epoch-{}-batch'.format(n_epoch, batch_size)

    lstmc = LSTMC(trainX.shape[1], time_step=5, n_class=2)

    if os.path.exists(lstmc_path):
        lstmc.load_state_dict(torch.load(lstmc_path))
    else:
        ctrain_result = lstmc.fit(trainX.copy(), trainy.copy(), n_epoch=n_epoch, batch_size=50, real_close=ttrainY)
        print(ctrain_result)

        torch.save(lstmc.state_dict(), lstmc_path)

    ctest_result = lstmc.predict(testX.copy(), testy.copy(), batch_size=1, real_close=ttestY)

    print(ctest_result)

    print(sohu.performance(test_result['target'], test_result['predicted'], True, ctest_result['predicted'], True))

