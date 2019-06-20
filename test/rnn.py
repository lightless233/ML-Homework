#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from torch import nn
import torch.utils.data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


EPOCH = 2
BATCH_SIZE = 64
# 多少个时间点的数据
TIME_STEP = 28
# 每个时间点的输入数据个数
# 图片是28*28的，所以输入28次，每次28个数据
INPUT_SIZE = 28
LR = 0.001
DOWNLOAD_MNIST = False


train_data = datasets.MNIST(
    root="./mnist/",
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)


test_data = datasets.MNIST(root="./mnist/", train=False, transform=transforms.ToTensor())
test_x = test_data.data.type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        _out, (hn, hc) = self.rnn(x, None)
        out = self.out(_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)


optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        bx = x.view(-1, 28, 28)
        output = rnn(bx)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)  # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            # accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)


# print 10 predictions from test data
test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
