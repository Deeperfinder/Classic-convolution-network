import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlockLow(nn.Module):    # 继承自nn.Module
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlockLow, self).__init__()            # self代表实例对象b，实例对象b通过nn.Module类调用方法__init__
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),                 # 改变输入数据
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride !=1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Resnet18(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(Resnet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride =2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.lastprocess = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] *(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.lastprocess(out)
        return out


def ResNet_18():

    return Resnet18(ResidualBlockLow)

# if __name__ == '__main__' :
#     X = torch.rand(1,3,224,224)
#     net = nn.Sequential(
#         ResNet_18().conv1,
#         ResNet_18().layer1,
#         ResNet_18().layer2,
#         ResNet_18().layer3,
#         ResNet_18().layer4,
#         ResNet_18().lastprocess
#     )
#     for layer in net:
#         X = layer(X)
#         print(layer.__class__.__name__, 'output shape:\t', X.shape)







