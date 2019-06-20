#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)


def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1),
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()

    for t in range(10000):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # entire net
    torch.save(net1, "net.pkl")
    # parameters
    torch.save(net1.state_dict(), "net_state.pkl")

    prediction = net1(x)

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title("NET1")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), "r-", lw=5)


def restore_net():
    net2 = torch.load("net.pkl")

    prediction = net2(x)

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(132)
    plt.title("NET2")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), "r-", lw=5)


def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 1),
    )
    net3.load_state_dict(torch.load("net_state.pkl"))

    prediction = net3(x)

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(133)
    plt.title("NET3")
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), "r-", lw=5)
    plt.show()


save()
restore_net()
restore_params()
