import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision import models

class Fire(nn.Module):
    def __init__(self, in_channel, s1x1, e1x1, e3x3):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channel, s1x1, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(s1x1, e1x1, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(s1x1, e3x3, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNet(nn.Module):
    def __init__(self, version='1_0', num_classes=640):
        super(SqueezeNet, self).__init__()
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=5, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=1, ceil_mode=False),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(512, 64, 256, 256),
            )
        else:
            # For SqueezeNet 1.1 or any other versions, you can modify here.
            raise ValueError("Unsupported SqueezeNet version")

        # Final convolution is initialized differently from the rest
        # final_conv = nn.Conv2d(512, 512, kernel_size=1)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     final_conv,
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1, 1))
        # )

        self.conv_end = nn.Conv2d(512, 10, kernel_size=3, stride=1, padding=2, bias=False)
        self.bn_end = nn.BatchNorm2d(10)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         if m is final_conv:
        #             init.normal_(m.weight, mean=0.0, std=0.01)
        #         else:
        #             init.kaiming_uniform_(m.weight)
        #         if m.bias is not None:
        #             init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        # x = self.classifier(x)
        x = self.conv_end(x)
        x = self.bn_end(x)
        x= torch.sigmoid(x)
        x = x.permute(0,2,3,1)
        return x


def squeezenet(pretrained=False):
    net = SqueezeNet(version='1_0', num_classes=640)
    # inp = Variable(torch.randn(64,3,32,32))
    # out = net.forward(inp)
    # print(out.size())
    return net
