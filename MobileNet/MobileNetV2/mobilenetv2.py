import torch
import torch.nn as nn
import torchsummary


def ConvBNReLU(in_channel, out_channel, kernel_size=3, stride=1, groups=1):
    padding=0 if kernel_size==1 else 1
    return nn.Sequential(
        nn.Conv2d(
            in_channel, out_channel, kernel_size=kernel_size, stride=stride,padding=padding, groups=groups,bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expansion_ratio=6):
        super(InvertedResidual, self).__init__()
        hidden_channel = round(in_channel * expansion_ratio)
        self.identity = stride ==1 and in_channel==out_channel

        layers = []
        if expansion_ratio != 1:
           layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1)) # pointwise
        layers.extend([
            # depthwise
            ConvBNReLU(hidden_channel, hidden_channel, stride = stride, groups=hidden_channel),
            # pointwise
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.identity:
            return x+self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, ):
        super(MobileNetV2, self).__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),
            InvertedResidual(32, 16, stride=1, expansion_ratio=1),
            InvertedResidual(16, 24, stride=2),
            InvertedResidual(24, 24, stride=1),
            InvertedResidual(24, 32, stride=2),
            InvertedResidual(32, 32, stride=1),
            InvertedResidual(32, 32, stride=1),
            InvertedResidual(32, 64, stride=2),
            InvertedResidual(64, 64, stride=1),
            InvertedResidual(64, 64, stride=1),
            InvertedResidual(64, 64, stride=1),
            InvertedResidual(64, 96, stride=1),
            InvertedResidual(96, 96, stride=1),
            InvertedResidual(96, 96, stride=1),
            InvertedResidual(96, 160, stride=2),
            InvertedResidual(160, 160, stride=1),
            InvertedResidual(160, 160, stride=1),
            InvertedResidual(160, 320, stride=1),
            ConvBNReLU(320, 1280, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1280, num_classes),
            # nn.Softmax(dim=1)     #(n,10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# testnet = MobileNetV2(10).to(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# torchsummary.summary(testnet, (3,224,224))












