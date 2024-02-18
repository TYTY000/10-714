import sys

from numpy.random import shuffle

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            norm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, dim),
            norm(dim)
            )
    return nn.Sequential(nn.Residual(modules), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    modules = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for _ in range(num_blocks):
        modules.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    hit, tango = 0, 0
    loss_func = nn.SoftmaxLoss()
    miss = 0
    if opt is not None:
        model.train()
        for i, data in enumerate(dataloader):
            x, y = data
            output = model(x)
            opt.reset_grad()
            loss = loss_func(output, y)
            miss += loss.numpy()
            loss.backward()
            opt.step()
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            tango += y.shape[0]
    else:
        model.eval()
        for i, data in enumerate(dataloader):
            x, y = data
            output = model(x)
            loss = loss_func(output, y)
            miss += loss.numpy()
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            tango += y.shape[0]
    err = (tango - hit) / tango
    return err, miss / (i + 1)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_data = ndl.data.MNISTDataset(
        data_dir + '/train-images-idx3-ubyte.gz',
        data_dir + '/train-labels-idx1-ubyte.gz'
    )
    test_data = ndl.data.MNISTDataset(
        data_dir + '/t10k-images-idx3-ubyte.gz',
        data_dir + '/t10k-labels-idx1-ubyte.gz',
    )
    train_loader = ndl.data.DataLoader(train_data, batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_data, batch_size)
    model = MLPResNet(784, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        train_acc, train_miss = epoch(train_loader, model, opt)
        print(f"{train_acc} {train_miss}\n")
    test_acc, test_miss = epoch(test_loader, model)
    print(f"{test_acc} {test_miss}\n")
    return (train_acc, train_miss, test_acc, test_miss)

    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
