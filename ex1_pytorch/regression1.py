#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy
import torch.nn
import matplotlib.pyplot as plt

INPUT_SIZE = 1
OUTPUT_SIZE = 1
LEARN_RATE = 0.01
MAX_EPOCHS = 1000


def load(filename):
    _x = []
    _y = []

    with open(filename, "r") as fp:
        for line in fp:
            line = line.strip()
            x, y = line.split(",")
            _x.append(float(x.strip()))
            _y.append(float(y.strip()))

    return _x, _y


def main():
    x, y = load("ex1data1.txt")
    x = [[item] for item in x]
    y = [[item] for item in y]
    x = numpy.array(x, dtype=numpy.float32)
    y = numpy.array(y, dtype=numpy.float32)

    model = torch.nn.Linear(INPUT_SIZE, OUTPUT_SIZE)
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
    plt.plot(x, y, 'ro', label='Original data')
    plt.plot(x, predicted, label='Fitted line')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
