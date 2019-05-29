#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch.nn
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

INPUT_SIZE = 2
OUTPUT_SIZE = 1
LEARN_RATE = 0.01
MAX_EPOCHS = 1000


def load(filename):
    x = []
    y = []

    with open(filename, "r") as fp:
        for line in fp:
            line = line.strip().split(",")

            x.append(line[:len(line) - 1])
            y.append([line[-1]])

    return x, y


def feature_scaling(data):
    mean_value = np.mean(data, axis=0)
    std_value = np.std(data, axis=0)
    return (data - mean_value) / std_value, mean_value, std_value


def main():
    x, y = load("ex1data2.txt")
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    x, mean_x, std_x = feature_scaling(x)
    y, mean_y, std_y = feature_scaling(y)

    model = torch.nn.Linear(2, 1)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARN_RATE)

    inputs = torch.from_numpy(x)
    targets = torch.from_numpy(y)

    for epoch in range(MAX_EPOCHS):
        outputs = model(inputs)
        loss = loss_func(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, MAX_EPOCHS, loss.item()))

    predicted = model(inputs).detach().numpy()
    print(predicted)

    x1, x2 = x[:, 0], x[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x1, x2, y, label="origin data")
    ax.plot_trisurf(x1, x2, predicted[:, 0], label="predicted")
    plt.show()

    def rotate(angle):
        ax.view_init(azim=angle)

    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
    rot_animation.save('ex1data2.gif', dpi=80, writer='imagemagick')


if __name__ == '__main__':
    main()
