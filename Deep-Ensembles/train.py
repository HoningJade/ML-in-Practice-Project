import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.activation = nn.ReLU()
        self.input = nn.Linear(4, 32)
        self.hidden = nn.Linear(32, 32)
        self.output = nn.Linear(32, 2)

    def forward(self, x):
        x = self.input(x)
        x = self.activation(x)
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x


class DonorsChooseDataset(Dataset):
    def __init__(self, phase='train', period='recent'):
        super(DonorsChooseDataset, self).__init__()
        x_fpath = os.path.join('./data', f'{phase}_X_{period}.csv')
        y_fpath = os.path.join('./data', f'{phase}_Y_{period}.npy')
        self.X = np.loadtxt(x_fpath, delimiter=",", dtype=str)
        self.X = self.X[1:]
        self.X = self.X[:, 1:].astype(float) # (len, 32)

        # normalize X
        min_ = torch.min(torch.tensor(self.X), 0, keepdim=True)
        max_ = torch.max(torch.tensor(self.X), 0, keepdim=True)
        self.X = (torch.tensor(self.X) - min_.values) / (max_.values - min_.values + 10e-12)

        self.Y = np.load(y_fpath).astype(int) # (len,)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        X = self.X[idx]
        Y = self.Y[idx]
        return (X, Y)


def accuracy(loader):
    correct_cnt = 0
    for i, (xs, ys) in enumerate(loader):
        xs = xs.to(device).float()
        ys = ys.to(device)
        yhat = model(xs)
        yhat = F.softmax(yhat, dim=1)
        yhat = torch.argmax(yhat, dim=1)
        correct_cnt += sum(ys == yhat).cpu().item()
    return correct_cnt / len(loader)


def accuracy_ensemble(loader, esize, t=0.5):
    idx = np.random.randint(1, high=16, size=esize)
    models = []
    for id in idx:
        model = NeuralNetwork()
        model.load_state_dict(torch.load(f'./model{id}.pt'))
        model.to(device)
        models.append(model)
    correct_cnt = 0
    for i, (xs, ys) in enumerate(loader):
        xs = xs.to(device).float()
        ys = ys.to(device)
        yhat_all = []
        for model in models:
            yhat = model(xs)
            yhat = F.softmax(yhat, dim=1)
            yhat_all.append(yhat)
        yhat_all = torch.cat(yhat_all, dim=0)
        yhat_all = torch.mean(yhat_all, dim=0)
        if t==0.5:
            yhat = torch.argmax(yhat_all, dim=1)
        else:
            yhat = torch.tensor([yhat_all > t])
        correct_cnt += sum(ys == yhat).cpu().item()
    return correct_cnt / len(loader)


def entropy_cnt(loader):
    count = {
        '0.1': 0,
        '0.2': 0,
        '0.3': 0,
        '0.4': 0,
        '0.5': 0,
        '0.6': 0,
        '0.7': 0,
    }
    for i, (xs, ys) in enumerate(loader):
        xs = xs.to(device).float()
        ys = ys.to(device)
        yhat = model(xs)
        loss = CELoss(yhat, ys)
        if 0 < loss.cpu().item() <= 0.1:
            count['0.1'] += 1
        elif 0.1 < loss.cpu().item() <= 0.2:
            count['0.2'] += 1
        elif 0.2 < loss.cpu().item() <= 0.3:
            count['0.3'] += 1
        elif 0.3 < loss.cpu().item() <= 0.4:
            count['0.4'] += 1
        elif 0.4 < loss.cpu().item() <= 0.5:
            count['0.5'] += 1
        elif 0.5 < loss.cpu().item() <= 0.6:
            count['0.6'] += 1
        else:
            count['0.7'] += 1
    return count


if __name__ == '__main__':

    # X = np.loadtxt('./data/train_X_old.csv', delimiter=",", dtype=str)
    # X = X[1:]
    # X = X[:, 1:].astype(float)  # (len, 32)
    # print(X)

    for idx in range(1, 16):
        exp_name = f'model{idx}'

        writer = SummaryWriter(log_dir=f'./runs/{exp_name}')

        lr = 0.01
        epochs = 100
        batchsize = 32

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train = DonorsChooseDataset(phase='train', period='recent')
        train_loader = DataLoader(train, batchsize, num_workers=4)
        test = DonorsChooseDataset(phase='val', period='recent')
        test_loader = DataLoader(test, batchsize, num_workers=4)
        print("Number of training and testing: %i, %i" % (len(train), len(test)))

        model = NeuralNetwork().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        CELoss = torch.nn.CrossEntropyLoss()

        results = []
        for i in range(epochs):
            epoch = i + 1
            model.train()
            loss_total = 0
            for i, (xs, ys) in enumerate(train_loader):
                xs = xs.to(device).float()
                ys = ys.to(device)
                yhat = model(xs)
                loss = CELoss(yhat, ys)
                loss_total += loss.cpu().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_acc = accuracy(train_loader)
            val_acc = accuracy(test_loader)
            print(f'Epoch: {epoch}, Train Acc: {train_acc}, Val Acc: {val_acc} Train Loss: {loss_total / len(train_loader)}')
            writer.add_scalars('acc', {'train_acc': train_acc, 'val_acc': val_acc}, epoch)

        torch.save(model.state_dict(), f"model{exp_name}.pt")
        writer.flush()

    train_acc = []
    val_acc = []
    for esize in range(1, 16):
        train_acc.append(accuracy_ensemble(train_loader, esize))
        val_acc.append(accuracy_ensemble(test_loader, esize))
    with open('train_acc.npy', 'wb') as f:
        np.save(f, train_acc)
    f.close()
    with open('val_acc.npy', 'wb') as f:
        np.save(f, val_acc)
    f.close()

    for esize in [1, 5, 10]:
        train_acc = []
        for threshold in np.arange(0.1, 1.1, 0.1):
            train_acc.append(accuracy_ensemble(train_loader, esize, t=threshold))
        with open(f'esize_{esize}_acc.npy', 'wb') as f:
            np.save(f, train_acc)
        f.close()


    for period in ['old', 'recent']:
        val = DonorsChooseDataset(phase='train', period=period)
        val_loader = DataLoader(val, 1, num_workers=4)
        loss = entropy_cnt(val_loader)
        loss_cnt = []
        for (k,v) in loss.items():
            loss_cnt.append(v)
        with open(f'entropy_{period}.npy', 'wb') as f:
            np.save(f, loss_cnt)
        f.close()