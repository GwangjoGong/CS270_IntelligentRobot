from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class Custom_Network(nn.Module):
    def __init__(self):
        # super() function makes class inheritance more manageable and extensible
        super(Custom_Network, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6,
                      kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=19,
                      kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=19 * 29 * 29, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=2)
        )

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)

        x = torch.flatten(x, 1)
        # print(x.shape)

        x = self.fc(x)

        output = F.log_softmax(x, dim=1)
        return output
