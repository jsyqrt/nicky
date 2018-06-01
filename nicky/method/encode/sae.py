# coding: utf-8
# Stacked AutoEncoder

import numpy as np 
import sklearn

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

np.random.seed(0)
torch.manual_seed(0)


class Autoencoder(torch.nn.Module):
    def __init__(self, n_in, n_hidden=10, sparsity_target=0.05, sparsity_weight=0.2, lr=0.0001, weight_decay=0.0):
        super(Autoencoder, self).__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.weight_decay = weight_decay
        self.lr = lr
        self.build_model()

    def build_model(self):
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.n_in, self.n_hidden),
            torch.nn.Sigmoid()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.n_hidden, self.n_in),
            torch.nn.Sigmoid(),
        )
        self.l1_loss = torch.nn.L1Loss(size_average=False)
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr, weight_decay=self.weight_decay, )
    
    def forward(self, inputs):
        hidden = self.encoder(inputs)
        hidden_mean = torch.mean(hidden, dim=0)
        sparsity_loss = torch.sum(self.kl_divergence(self.sparsity_target, hidden_mean))
        return self.decoder(hidden), sparsity_loss

    def kl_divergence(self, p, q):
        # Kullback Leibler divergence
        return p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))

    def fit(self, X, n_epoch=10, batch_size=64, en_shuffle=True):
        for epoch in range(n_epoch+1):
            if en_shuffle:
                X = sklearn.utils.shuffle(X)
            for local_step, X_batch in enumerate(self.gen_batch(X, batch_size)):
                inputs = torch.autograd.Variable(torch.from_numpy(X_batch.astype(np.float32)))
                outputs, sparsity_loss = self.forward(inputs)

                l1_loss = self.l1_loss(outputs, inputs)
                loss = l1_loss + self.sparsity_weight * sparsity_loss                   
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if epoch % (n_epoch//3) == 0:
                print ("Epoch %d/%d | train loss: %.4f | l1 loss: %.4f | sparsity loss: %.4f"
                        %(epoch+1, n_epoch, loss.data[0], l1_loss.data[0], sparsity_loss.data[0]))
                #'''

    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i : i+batch_size]


class StackedAutoencoder():
    def __init__(self, dims, sparsity_target=0.05, sparsity_weight=0.2, lr=0.0001, weight_decay=0.0):
        self.layers = {}
        self.dims = dims
        for i, dim in enumerate(self.dims[:-1]):
            self.layers[i] = Autoencoder(dim, self.dims[i+1], sparsity_target, sparsity_weight, lr, weight_decay)
        self.outs = {}
    
    def train(self, x, n_epoch=15000):
        self.outs[0] = torch.autograd.Variable(torch.from_numpy(x.astype(np.float32)))
        for i, _ in enumerate(self.dims[:-1]):
            self.outs[i] = self.outs[i].data.numpy()
            self.layers[i].fit(self.outs[i], n_epoch=n_epoch)
            self.outs[i] = torch.autograd.Variable(torch.from_numpy(self.outs[i].astype(np.float32)))
            self.outs[i+1] = self.layers[i].encoder(self.outs[i])
    
    def transform(self, x):
        x = torch.autograd.Variable(torch.from_numpy(x.astype(np.float32)))
        for i, _ in enumerate(self.dims[:-1]):
            self.layers[i].eval()
            x = self.layers[i].encoder(x)
        return x.data.numpy()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    from stocks.data.sohu import sohu

    data = sohu.get_hist_data('zs_000001', indicators=True, preprocess=True)
    data = data.fillna(data.mean().mean()).values
    sae = StackedAutoencoder([data.shape[1], 10, 10, 10, 10], )
    sae.train(data)
    transformed = sae.transform(data)
