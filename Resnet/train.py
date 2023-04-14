import torch
from resnet import *
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

max_epoch = 50
pre_epoch = 0
batch_size = 128
best_acc = 0.85

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), # 四周填充0，图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 训练集
train_set = torchvision.datasets.CIFAR10(
    root='./', train=True, download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size = batch_size, shuffle=True, num_workers=0)

# 测试集
test_set = torchvision.datasets.CIFAR10(
    root = './', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size = batch_size, shuffle=True, num_workers=0)


model = ResNet_18().to(device)
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    with open('text/acc.txt', 'w') as f:
        with open('text/log.txt', 'w') as f2:
            for epoch in range(pre_epoch, max_epoch):
                optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay=5e-4)
                #ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
                CosLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
                model.train()
                train_loss = []
                train_accs = []
                for i,data in enumerate(train_loader,0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad() # 每个batch的梯度需要清零
                    loss.backward()
                    optimizer.step()

                    acc = (outputs.argmax(dim=-1) == labels.to(device)).float().mean()
                    train_loss.append(loss.item())
                    train_accs.append(acc)

                    if i % 20 == 0:
                        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                              % (epoch + 1, (i + 1 + epoch * len(train_loader)), sum(train_loss) / len(train_loss),
                                 100. * sum(train_accs) / len(train_accs)))

                        f2.write(
                            '[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                            % (epoch + 1, (i + 1 + epoch * len(train_loader)), sum(train_loss) / len(train_loss),
                               100. * sum(train_accs) / len(train_accs))
                        )
                        f2.write('\n')
                        f2.flush()
                print("**************************Waiting Test!****************************")
                model.eval()
                valid_loss = []
                valid_accs = []
                for data in test_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        outputs = model(images)

                    loss = criterion(outputs, labels.to(device))
                    acc = (outputs.argmax(dim=-1) == labels.to(device)).float().mean()
                    valid_loss.append(loss.item())
                    valid_accs.append(acc)
                valid_loss = sum(valid_loss) / len(valid_loss)
                valid_acc = sum(valid_accs) / len(valid_accs)
                print(f"[ Valid | {epoch + 1:03d}/{max_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

                torch.save(model, 'model/net_%03d.pth' % (epoch + 1))
                f.write("epoch=%03d,Accuracy= %.3f%%" % (epoch + 1, valid_acc))
                f.write('\n')
                f.flush()

                if valid_acc > best_acc:
                    f3 = open("text/best_acc.txt", "w")
                    f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, valid_acc))
                    f3.close()
                    best_acc = valid_acc







