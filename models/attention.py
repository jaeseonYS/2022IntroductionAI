import torch
from torch import nn
from models.mlp import MLP
from models.lstm import LSTM
from models.cnn import CNN
from models.lstmfcn import LSTMFCN


class Attention(nn.Module):
    def __init__(self, in_channel, hid_channel, out_channel):
        super().__init__()

        self.mlp = MLP(in_channel, hid_channel, out_channel)
        self.lstm = LSTM(in_channel, hid_channel, out_channel)
        self.cnn = CNN(1, hid_channel, out_channel)
        self.lstmfcn = LSTMFCN(in_channel, hid_channel, out_channel)

        self.w_q = nn.Linear(hid_channel, hid_channel)
        self.w_k = nn.Linear(hid_channel, hid_channel)
        self.w_v = nn.Linear(hid_channel, hid_channel)

        self.softmax = nn.Softmax()
        self.fc = nn.Linear(hid_channel, out_channel)

    def forward(self, x, view=False):
        f1, _ = self.mlp(x)
        f2, _ = self.lstm(x)
        f3, _ = self.cnn(x)
        f4, _ = self.lstmfcn(x)
        z = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1), f3.unsqueeze(1), f4.unsqueeze(1)], dim=1)

        # attention
        q = self.w_q(z)
        k = self.w_k(z)
        v = self.w_v(z)
        score = self.softmax(q @ k.transpose(2, 1))
        x = score @ v
        x = torch.sum(x, dim=1)

        if view:
            return z, x, score

        out = self.softmax(self.fc(x))
        return x, out
