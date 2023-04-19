import torch
import torch.nn as nn
import torchsummary

def _conv_block(in_channel, out_channel, stride):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def _neck_block(in_channel, out_channel, num_layers, stride=1,):
    layers = []
    for i in range(num_layers):
        layers.append(_dw_conv_block(in_channel, out_channel, stride=stride))
    return nn.Sequential(*layers)      # 此处*的作用为解包，关键字参数传入函数中

def _dw_conv_block(in_channel, out_channel, stride):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_channel), # 分组卷积
        nn.BatchNorm2d(in_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )


class MobileNetV1(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV1, self).__init__()
        self.mobilebone = nn.Sequential(
            _conv_block(3,32,2),
            _dw_conv_block(32, 64, 1),
            _dw_conv_block(64, 128, 2),
            _dw_conv_block(128, 128, 1),
            _dw_conv_block(128, 256, 2),
            _dw_conv_block(256, 256, 1),
            _dw_conv_block(256, 512, 2),
            _neck_block(512, 512, num_layers=5),
            _dw_conv_block(512, 1024, 2),
            _dw_conv_block(1024, 1024, 1),
            nn.AvgPool2d(kernel_size=7, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.mobilebone(x)
        x = torch.flatten(x, 1) # (n,1024,1,1) -> (n, 1024)
        return self.classifier(x)




# testnet = MobileNetV1(10).to(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# torchsummary.summary(testnet, (3,224,224))


