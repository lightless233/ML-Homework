#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from matplotlib import pyplot


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


def scaling(data):
    """
    缩放特征
    """

    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    print("data mean: {}\ndata_std:{}".format(data_mean, data_std))

    if len(data.shape) > 1:
        for idx in range(data.shape[1]):
            column = data[:, idx]
            column = (column - data_mean[idx]) / data_std[idx]
            data[:, idx] = column
    else:
        column = (data - data_mean) / data_std
        data = column

    return data, data_mean, data_std


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
    y, y_mean, y_std = scaling(y)

    # 补上偏置
    bias = np.ones((x.shape[0], 1))
    x = np.column_stack([bias, x])
    print(x)

    theta = np.zeros([x.shape[1], 1])
    print(cost_function(theta, x, y))

    # 正规方程
    theta = normal_equations(x, y)
    print("theta is:", theta)

    print(cost_function(theta, x, y))


if __name__ == '__main__':
    main()
