import torch
from torch import nn
import torchsummary
class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()
        self.net = nn.Sequential(
            # 这里使用11*11来捕捉对象
            nn.Conv2d(3, 96, kernel_size=(11,11), stride=4,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=(5,5), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,10)
        )
    def forward(self, x):
        out = self.net(x)
        return out

print(Alexnet())
testnet = Alexnet().to(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
torchsummary.summary(testnet, (3,224,224))