import os
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils import data
from scipy import io
import spd.nn as nn_spd
import time
from spd.optimizers import MixOptimizer



class signal2spd(nn.Module):
    # convert signal epoch to SPD matrix
    def __init__(self):
        super().__init__()
        self.dev = th.device('cpu')

    def forward(self, x):
        x = x.squeeze()
        mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])

        # Compute the covariance matrix
        x = x - mean
        cov = x @ x.permute(0, 2, 1)  # 即 x * x.T
        cov = cov.to(self.dev)
        cov = cov / (x.shape[-1] - 1)

        # Ensure all elements on the main diagonal are greater than 0
        identity = th.eye(cov.shape[-1], cov.shape[-1], device=self.dev).to(self.dev).repeat(x.shape[0], 1, 1)
        cov = cov + (1e-5 * identity)
        return cov


class E2R(nn.Module):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.signal2spd = signal2spd()

    def patch_len(self, n, epochs):
        """Divide the feature vectors into a given number of epochs and return a list,
        where each element represents the number of feature vectors in that epoch."""
        list_len = []
        base = n // epochs
        for i in range(epochs):
            list_len.append(base)
        for i in range(n - base * epochs):
            list_len[i] += 1
        # 验证
        if sum(list_len) == n:
            return list_len
        else:
            return ValueError('check your epochs and axis should be split again')

    def forward(self, x):
        # x with shape[bs, ch, time] batch size, channels, time
        # split feature into several epochs.
        list_patch = self.patch_len(x.shape[-1], int(self.epochs))
        x_list = list(th.split(x, list_patch, dim=-1))
        # convert each one to a specific SPD matrix.
        for i, item in enumerate(x_list):
            x_list[i] = self.signal2spd(item)

        # Change the data shape from [epoch, bs, ...] to [bs, epoch, ...]
        x = th.stack(x_list).permute(1, 0, 2, 3)
        return x


class mamemNet(nn.Module):
    def __init__(self):
        super(__class__, self).__init__()
        dim1 = 12
        classes = 5
        self.conv1 = nn.Conv2d(1, 125, (8, 1))
        self.Bn1 = nn.BatchNorm2d(125)
        self.conv2 = nn.Conv2d(125, 15, (1, 36), padding=(0, 18))
        self.Bn2 = nn.BatchNorm2d(15)
        self.ract1 = E2R(1)
        self.re = nn_spd.ReEig()
        self.bimap1 = nn_spd.BiMap(1, 1, 15, dim1)
        self.batchnorm1 = nn_spd.BatchNormSPD(dim1)
        self.logeig = nn_spd.LogEig()
        self.linear = nn.Linear(dim1 ** 2, classes).double()

    def forward(self, x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.conv2(x)
        x = self.Bn2(x)
        x = self.ract1(x.double())

        x_spd = self.bimap1(x.double())
        x_spd = self.batchnorm1(x_spd)
        x_spd = self.re(x_spd)

        x_vec = self.logeig(x_spd).view(x_spd.shape[0], -1)
        y = self.linear(x_vec)
        return y


def mamem(data_loader):
    # main parameters
    lr = 2.5e-3  # learning rat
    epochs = 200
    model_path = './checkpoint/mamem/'
    # setup data and model

    model = mamemNet()

    # setup loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    opti = MixOptimizer(model.parameters(), lr=lr, momentum=.9, weight_decay=5e-2)
    bestLoss = 1e10
    # training loop
    best = 0
    loss_val, acc_val = [], []
    for epoch in range(epochs):

        loss_val, acc_val = [], []
        y_true, y_pred = [], []
        gen = data_loader._val_generator

        # train one epoch
        ST = time.time()
        loss_train, acc_train = [], []
        model.train()
        for local_batch, local_labels in data_loader._train_generator:
            opti.zero_grad()
            out = model(local_batch)
            l = loss_fn(out, local_labels)
            acc, loss = (out.argmax(1) == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
            loss_train.append(loss)
            acc_train.append(acc)
            l.backward()
            opti.step()
        ET = time.time()
        UT = ET - ST
        print(UT)
        acc_train = np.asarray(acc_train).mean()
        loss_train = np.asarray(loss_train).mean()
        print('train loss: ' + str(loss_train) + '; train acc: ' + str(100 * acc_train) + '% at epoch ' + str(
            epoch + 1) + '/' + str(epochs))

        # validation
        model.eval()
        for local_batch, local_labels in gen:
            out = model(local_batch)
            l = loss_fn(out, local_labels)
            predicted_labels = out.argmax(1)
            y_true.extend(list(local_labels.cpu().numpy()))
            y_pred.extend(list(predicted_labels.cpu().numpy()))
            acc, loss = (predicted_labels == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
            loss_val.append(loss)
            acc_val.append(acc)
        acc_val = np.asarray(acc_val).mean()
        loss_val = np.asarray(loss_val).mean()
        if loss_val < bestLoss:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            bestLoss = loss_val
            final_path = os.path.join(model_path, f'best_model_lr{lr}2.pt')
            th.save(model, final_path)
        print('Val acc: ' + str(100 * acc_val) + '% at epoch ' + str(epoch + 1) + '/' + str(epochs))

    # test accuracy
    loss_test, acc_test = [], []
    y_true, y_pred = [], []
    gen = data_loader._test_generator
    model.eval()
    for local_batch, local_labels in gen:
        out = model(local_batch)
        l = loss_fn(out, local_labels)
        predicted_labels = out.argmax(1)
        y_true.extend(list(local_labels.cpu().numpy()))
        y_pred.extend(list(predicted_labels.cpu().numpy()))
        acc, loss = (predicted_labels == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
        loss_test.append(loss)
        acc_test.append(acc)
    acc_test = np.asarray(acc_test).mean()
    loss_test = np.asarray(loss_test).mean()
    log_file.write('%f\n' % acc_test)
    log_file.flush()
    all_acc.append(acc_test)
    print('Final validation accuracy: ' + str(100 * acc_val) + '%')
    print('Test accuracy: ' + str(100 * acc_test) + '%')
    if len(all_acc) == 11:
        mean_acc = np.mean(all_acc)
        std_acc = np.std(all_acc)
        print('the acc of all subjects is: ' + str(100 * mean_acc) + '%')
        print('the std of all subjects is: ' + str(std_acc))
    return 100 * acc_test


if __name__ == "__main__":
    data_path = 'data/MAMEM/'  # data path
    pval = 0.25  # validation percentage
    ptest = 0.25  # test percentage
    batch_size = 30  # batch size


    class DatasetRadar(data.Dataset):
        def __init__(self, path, names):
            self._path = path
            self._names = names

        def __len__(self):
            return len(self._names)

        def __getitem__(self, item):
            x = np.load(self._path + self._names[item])
            x = np.concatenate((x.real[:, None], x.imag[:, None]), axis=1).T
            x = th.from_numpy(x)
            y = int(self._names[item].split('.')[0].split('_')[-1])
            y = th.from_numpy(np.array(y))
            return x.float(), y.long()


    class DataLoaderRadar:
        def __init__(self, data_path, subject, batch_size):
            dev = th.device("cpu")
            train = io.loadmat(os.path.join(data_path, 'U' + f'{subject:03d}' + '.mat'))
            tempdata = th.Tensor(train['x_test']).unsqueeze(1)
            templabel = th.Tensor(train['y_test']).view(-1)
            # 划分训练集，验证集，测试集
            x_train = tempdata[:300]
            y_train = templabel[:300]

            x_valid = tempdata[300:400]
            y_valid = templabel[300:400]

            x_test = tempdata[400:500]
            y_test = templabel[400:500]

            x_train = x_train.to(dev)
            y_train = y_train.long().to(dev)
            x_valid = x_valid.to(dev)
            y_valid = y_valid.long().to(dev)
            x_test = x_test.to(dev)
            y_test = y_test.long().to(dev)

            print(x_train.shape)
            print(y_train.shape)
            print(x_valid.shape)
            print(y_valid.shape)
            print(x_test.shape)
            print(y_test.shape)

            train_dataset = data.TensorDataset(x_train, y_train)
            valid_dataset = data.TensorDataset(x_valid, y_valid)
            test_dataset = data.TensorDataset(x_test, y_test)

            self._train_generator = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # true
            self._test_generator = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
            self._val_generator = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    kk = list(range(1, 12))
    log_file = open('mamem.txt', 'a')
    res = th.zeros(5, 11)
    all_acc = []
    for j in range(5):
        for i in kk:
            res[j, i - 1] = mamem(DataLoaderRadar(data_path, kk[i - 1], batch_size))  # kk[i-1]
    print(res)
    log_file.close()

