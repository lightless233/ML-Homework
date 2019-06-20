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
    """
    loss = np.dot(theta.T, x_data) - y_data
    tmp = np.power(loss, 2)
    result = tmp.sum() / m / 2
    return result


def update_theta(theta, x_data, y_data, m):
    """
    批量梯度下降更新 theta 值
    """

    for i in range(ITERATIONS):
        h = np.dot(theta.T, x_data)
        loss = h - y_data
        t = loss * x_data
        delta = LEARN_ALPHA * np.mean(t, axis=1)

        theta = theta.T - delta
        theta = theta.T
        # print(theta)
        j_value = compute_cost_function(m, theta, x_data, y_data)
        print("j value: {}".format(j_value))

    return theta


def prediction(x, theta):
    return x, theta[0][0] + theta[1][0] * x


def main():
    x_data, y_data = load("ex1data1.txt")
    print(x_data, "\n", y_data)

    # 绘制原始数据的散点图
    plot.scatter(x_data, y_data)
    plot.xlabel("Population of City in 10,000s")
    plot.ylabel("Profit in $10,000s")
    # plot.show()
    plot.savefig("raw.jpg")

    # theta 初始值设置为0 （2x1的矩阵）
    theta = np.zeros((2, 1))

    # 获取训练集大小
    m = len(y_data)
    print("m: {}".format(m))

    # 生成X的数组，应该为 2×m 的矩阵
    x_data = np.row_stack([np.ones((m, 1)).T, x_data])
    print("x_data: {}".format(x_data))
    print("x_data ndim: {}".format(x_data.shape))

    # 计算初始的代价函数值
    # 应该为32.07
    print(compute_cost_function(m, theta, x_data, y_data))

    # 迭代进行梯度下降
    final_theta = update_theta(theta, x_data, y_data, m)
    print("final theta:", final_theta)

    # 选取两个点绘制假设函数
    x = np.linspace(3.5, 24)
    y = final_theta[0][0] + final_theta[1][0] * x

    plot.plot(x, y, color="red")
    plot.show()


if __name__ == '__main__':
    main()
