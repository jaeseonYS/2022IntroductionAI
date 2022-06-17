import torch
from torch import nn


class LSTMFCN(nn.Module):
    def __init__(self, in_channel, hid_channel, out_channel):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channel, hidden_size=in_channel, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.layer = nn.Linear(in_channel, hid_channel)
        self.bn = nn.BatchNorm1d(hid_channel)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 2))
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, hid_channel, kernel_size=(2, 1), stride=(1, 1))
        self.bn5 = nn.BatchNorm2d(hid_channel)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.dense = nn.Linear(hid_channel * 2, hid_channel)

        self.fc = nn.Linear(hid_channel, out_channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # LSTM
        x1 = x.transpose(2, 1)
        x1, _ = self.lstm(x1)
        x1 = self.dropout(x1)
        x1 = x1.transpose(1, 2)
        x1 = torch.mean(x1, dim=2)
        x1 = self.relu(self.bn(self.layer(x1)))

        # FCN
        if x.dim() == 3:
            x2 = x.unsqueeze(1)
        x2 = self.pool(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.pool(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.pool(self.relu(self.bn3(self.conv3(x2))))
        x2 = self.pool(self.relu(self.bn4(self.conv4(x2))))
        x2 = self.pool(self.relu(self.bn5(self.conv5(x2)))).squeeze(3).squeeze(2)

        x = torch.cat([x1, x2], dim=1)
        x = self.dense(x)
        out = self.softmax(self.fc(x))
        return x, out

