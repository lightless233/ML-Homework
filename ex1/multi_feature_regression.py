#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


def load_data(filename):
    """
    读文件，x数据直接补上 bias
    """
    x = []
    y = []

    with open(filename, "r") as fp:
        for line in fp:
            _x1, _x2, _y = line.split(",")
            x.append(list(map(float, [_x1, _x2])))
            y.append(float(_y))

    return x, y


def scaling(x):
    """
    缩放特征
    """

    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    print("x mean: {}\nx_std:{}".format(x_mean, x_std))

    for idx in range(x.shape[1]):
        column = x[:, idx]
        column = (column - x_mean[idx]) / x_std[idx]
        x[:, idx] = column

    return x, x_mean, x_std


def cost_function(theta, x, y):
    tmp = np.dot(x, theta) - y
    return np.mean(np.dot(tmp.T, tmp)) / 2


def normal_equations(x, y):
    t = np.linalg.inv(np.dot(x.T, x))
    return np.dot(np.dot(t, x.T), y)


def main():
    x, y = load_data("ex1data2.txt")
    x = np.array(x)
    y = np.array(y)
    print("x.shape: {}, y.shape: {}".format(x.shape, y.shape))

    # 特征缩放，并返回相关的数据，后续做预测的时候需要用到
    x, x_mean, x_std = scaling(x)

    # 补上偏置
    bias = np.ones((x.shape[0], 1))
    x = np.column_stack([bias, x])
    print(x)

    theta = np.zeros([x.shape[1], 1])
    print(cost_function(theta, x, y))

    # 正规方程
    theta = normal_equations(x, y)
    print(theta)

    print(cost_function(theta, x, y))


if __name__ == '__main__':
    main()
