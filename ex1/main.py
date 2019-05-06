#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plot

# 梯度下降迭代次数
ITERATIONS = 1500
# 学习速率
LEARN_ALPHA = 0.01


def load(filename):
    """
    读取文件中的数据
    :param filename:
    :return:
    """
    x_data = []
    y_data = []

    with open(filename, "r") as fp:
        for line in fp:
            line = line.strip()
            x, y = line.split(",")
            x_data.append(float(x.strip()))
            y_data.append(float(y.strip()))

    return x_data, y_data


def compute_cost_function(m, theta, x_data, y_data):
    """
    计算代价函数 J(theta) 的值
    :param m:
    :param theta:
    :param x_data:
    :param y_data:
    :return:
    """
    tmp = np.dot(theta.T, x_data.T) - y_data
    tmp = np.power(tmp, 2)
    result = tmp.sum() / m / 2
    # print(tmp)
    # print(result)
    return result


def update_theta(theta, x_data, y_data):
    """
    批量梯度下降更新 theta 值
    :return:
    """

    xa = range(ITERATIONS)
    ya = []

    for i in range(ITERATIONS):
        tmp = np.dot(theta.T, x_data.T) - y_data
        tmp = np.dot(tmp, x_data)
        result = tmp.sum() * LEARN_ALPHA / len(y_data)
        print("result: {}".format(result))

        theta = theta - result
        print("new theta: {}".format(theta))
        ya.append(theta)

    # print(ya)


def main():
    x_data, y_data = load("ex1data1.txt")
    print(x_data, "\n", y_data)

    # 绘制原始数据的散点图
    plot.scatter(x_data, y_data)
    plot.xlabel("Population of City in 10,000s")
    plot.ylabel("Profit in $10,000s")
    plot.show()

    # theta 初始值设置为0 （2x1的矩阵）
    theta = np.zeros((2, 1))

    # 获取训练集大小
    m = len(y_data)
    print("m: {}".format(m))

    # 生成X的数组，应该为 mx2的矩阵
    x_data = np.column_stack([np.ones((m, 1)), x_data])
    # print("x_data: {}".format(x_data))
    print("x_data ndim: {}".format(x_data.shape))

    # 计算初始的代价函数值
    # 应该为32.07
    print(compute_cost_function(m, theta, x_data, y_data))

    # 迭代进行梯度下降
    update_theta(theta, x_data, y_data)


if __name__ == '__main__':
    main()
