import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, in_channel, hid_channel, out_channel):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channel, hidden_size=in_channel, num_layers=2, batch_first=True)
        self.layer = nn.Linear(in_channel, hid_channel)
        self.bn = nn.BatchNorm1d(hid_channel)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hid_channel, out_channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(2, 1)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)

        x = torch.mean(x, dim=2)
        x = self.relu(self.bn(self.layer(x)))
        out = self.softmax(self.fc(x))
        return x, out

