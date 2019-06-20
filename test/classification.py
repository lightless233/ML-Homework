#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn
import torch.nn.functional
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)

x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

x, y = torch.autograd.Variable(x), torch.autograd.Variable(y)


# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()

        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden(x))
        x = self.predict(x)
        return x


net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 2),
)

net1 = Net(2, 100, 2)
print(net1)
print(net2)
#
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
# loss_func = torch.nn.CrossEntropyLoss()
#
# plt.ion()  # something about plotting
#
# for t in range(1000):
#     out = net(x)
#
#     loss = loss_func(out, y)
#     print("t:", t, " loss:", loss)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if t % 2 == 0:
#         # plot and show learning process
#         plt.cla()
#         prediction = torch.max(out, 1)[1]
#         pred_y = prediction.data.numpy()
#         target_y = y.data.numpy()
#         plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
#         accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
#         plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
#         plt.pause(0.1)
#
# plt.ioff()
# plt.show()

# out = net(x)
# # 过了一道 softmax 的激励函数后的最大概率才是预测值
# prediction = torch.max(torch.nn.functional.softmax(out), 1)[1]
# pred_y = prediction.data.numpy().squeeze()
# target_y = y.data.numpy()
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
# accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
# plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
# plt.pause(0.1)
