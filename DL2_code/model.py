from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv_Network(nn.Module):
    def __init__(self):
        # super() function makes class inheritance more manageable and extensible
        # super() function makes class inheritance more manageable and extensible
        super(Custom_Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.max1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=16 * 2 * 29, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max1(x)
        # print(x.shape)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.max1(x)
        # print(x.shape)

        x = torch.flatten(x, 1)
        # print(x.shape)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        output = F.log_softmax(x, dim=1)
        return output
