import torch
from torch import nn
from torchvision.models.resnet import resnet50


class Resnet50(nn.Module):
    def __init__(self, out_channel):
        super().__init__()

        resnet = resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.base = nn.Sequential(*modules)
        # for param in self.base.parameters():
        #     param.requires_grad = False
        self.fc = nn.Linear(2048, out_channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.cat([x, x, x], dim=1)
        x = self.base(x).squeeze()
        out = self.softmax(self.fc(x))
        return x, out
