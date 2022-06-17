from torch import nn


class CNN(nn.Module):
    def __init__(self, in_channel, hid_channel, out_channel):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, 8, kernel_size=(3, 3), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 2))
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, hid_channel, kernel_size=(2, 1), stride=(1, 1))
        self.bn5 = nn.BatchNorm2d(hid_channel)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc = nn.Linear(hid_channel, out_channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.pool(self.relu(self.bn5(self.conv5(x)))).squeeze(3).squeeze(2)
        out = self.softmax(self.fc(x))
        return x, out
