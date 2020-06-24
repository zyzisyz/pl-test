#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: nnet.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Wed Jun 24 16:37:35 2020
# ************************************************************************/

import torch
from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):

    def __init__(self, hparams):
        super(CNN, self).__init__()
        self.hparams = hparams

        # build nnet
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Linear(nn.Module):

    def __init__(self, hparams):
        super(Linear, self).__init__()
        self.hparams = hparams

        # build nnet
        self.l1 = nn.Linear(28*28, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        output = F.log_softmax(x, dim=1)
        return output
