from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear_Network(nn.Module):
    def __init__(self):
        # super() function makes class inheritance more manageable and extensible
        super(Linear_Network, self).__init__()
        self.fc1 = nn.Linear(784, 256, bias=True)
        self.fc2 = nn.Linear(256, 128, bias=True)
        self.fc3 = nn.Linear(128, 10, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x
