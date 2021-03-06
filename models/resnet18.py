import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNet(nn.Module):

    def __init__(self, depth=18):

        super(ResNet, self).__init__()

        assert depth == 18, 'only supports depth 18'
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        zero_init_residual = False

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block1 = self._make_layer(BasicBlock, 64, 2)
        self.block2 = self._make_layer(BasicBlock, 128, 2, stride=2, dilate=False)
        self.block3 = self._make_layer(BasicBlock, 256, 2, stride=2, dilate=False)
        self.block4 = self._make_layer(BasicBlock, 512, 2, stride=2, dilate=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 512

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block, planes, blocks=2, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_c1 = self.conv1(x)
        x_b1 = self.block1(x_c1)
        x_b2 = self.block2(x_b1)
        x_b3 = self.block3(x_b2)
        x_b4 = self.block4(x_b3)
        x_pool = self.pool(x_b4)
        return x_pool


class Resnet18(nn.Module):
    def __init__(self, out_channel):
        super().__init__()

        self.base = ResNet()
        self.fc = nn.Linear(512, out_channel)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.base(x).squeeze()
        out = self.softmax(self.fc(x))
        return x, out
