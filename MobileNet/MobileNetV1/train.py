import argparse
import torchvision
import torch.distributed as dist
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.optim as optim

from tqdm import tqdm
from mobilenetv1 import *
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler


# 使用CUDA_VISIBLE_DEVICES指定GPU --nproc_per_node =2 用2卡
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --use_mix_precision



# max_epoch = 50
# batch_size = 64
# best_acc = 0.85
#
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
# # DDP多卡训练
# # torch.distributed.init_process_group(backend='gloo', init_method='env://', rank=0, world_size=1)
#
# transform_train = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
# transform_test = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
# # 训练集
# train_set = torchvision.datasets.CIFAR10(
#     root='../../Alexnet', train=True, download=False, transform=transform_train)
# train_sample = torch.utils.data.distributed.DistributedSampler(train_set)
# train_loader = torch.utils.data.DataLoader(
#     train_set, batch_size = batch_size, shuffle=False, sampler=train_sample)#num_workers=0)
#
# # 测试集
# test_set = torchvision.datasets.CIFAR10(
#     root = '../../Alexnet', train=False, download=False, transform=transform_test)
# test_sample = torch.utils.data.distributed.DistributedSampler(test_set)
# test_loader = torch.utils.data.DataLoader(
#     test_set, batch_size = batch_size, shuffle=False, sampler=test_sample) #num_workers=0)
#
#
#
# model = MobileNetV1(num_classes=10)
# model = model.cuda(args.local_rank)
# model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
# criterion = nn.CrossEntropyLoss()



def evalutate(model, gpu, test_loader, rank):
    model.eval()
    size = torch.tensor(0.).to(device)
    correct = torch.tensor(0.).to(device)
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            size += images.shape[0]
            correct += (outputs.argmax(1)==labels).type(torch.float).sum()

    dist.reduce(size, 0, op = dist.ReduceOP.SUM)
    dist.reduce(correct, 0, op = dist.ReduceOP.SUM)
    if rank ==0:
        print('Evaluate accuracy is {:.2f}'.format(correct/size))

def train(gpu, args):
    # 训练函数中仅需要更改初始化方式即可。在ENV中只需要指定init_method='env://'。
    # TCP所需的关键参数模型会从环境变量中自动获取，环境变量可以在程序外部启动时设定，参考启动方式。
    dist.init_process_group(backend='nccl', init_method='env://', world_size=2)
    args.rank = dist.get_rank()

    model = MobileNetV1(num_classes=10)
    model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().to(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # 并行环境下，对于用到BN层的模型需要转换为同步BN层；
    # 用Distributed Dataparallel将模型封装为一个DDP模型，并复制到指定的GPU上
    # 封装时不需要改变模型内部的代码；设置混合精度中scaler，通过设置enabled参数控制是否生效
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids = [gpu])
    scaler = GradScaler(enabled=args.use_mix_precision)

    # Data loading
    # 训练集
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root='../../Alexnet', train=True, download=False, transform=transform_train)
    train_sample = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sample)  # num_workers=0)

    # 测试集
    test_set = torchvision.datasets.CIFAR10(
        root='../../Alexnet', train=False, download=False, transform=transform_test)
    test_sample = torch.utils.data.distributed.DistributedSampler(test_set)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True,sampler=test_sample)  # num_workers=0)

    total_step = len(train_loader)      #这里的意思是按batch_size的大小，总共有多少份

    for epoch in range(args.epochs):
        # 在每个epoch开始前打乱数据顺序
        train_loader.sampler.set_epoch(epoch)
        model.train()
        for i,(images,labels) in enumerate(tqdm(train_loader)):
            images= images.to(gpu)
            labels = labels.to(gpu)
            # 半精度训练
            with torch.cuda.amp.autocast(enabled=args.use_mix_precision):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # backward and optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 避免LOG信息重复打印，只允许rank0进程打印
            if(i+1)%100==0 and args.rank==0:
                print('Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                   loss.item()))
            dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')


    parser.add_argument('-g', '--gpuid', default=0, type=int, help = "which gpu to use")
    parser.add_argument('-e', '--epochs', default=50, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=64, type=int,
                        metavar='N',
                        help='number of batchsize')

    ##################################################################################
    # 这里指的是当前进程在当前机器中的序号，注意和在全部进程中序号的区别，即指的是GPU序号0,1,2,3。
    # 在ENV模式中，这个参数是必须的，由启动脚本自动划分，不需要手动指定。要善用local_rank来分配GPU_ID。
    # 不需要填写，脚本自动划分
    parser.add_argument("--local_rank", type=int,  #
                        help='rank in current node')  #
    # 是否使用混合精度
    parser.add_argument('--use_mix_precision', default=False,  #
                        action='store_true', help="whether to use mix precision")  #
    # Need 每台机器使用几个进程，即使用几个gpu  双卡2，
    parser.add_argument("--nproc_per_node", type=int,  #
                        help='numbers of gpus')  #
    # 分布式训练使用几台机器，设置默认1，单机多卡训练
    parser.add_argument("--nnodes", type=int, default=1, help='numbers of machines')
    # 分布式训练使用的当前机器序号，设置默认0，单机多卡训练只能设置为0
    parser.add_argument("--node_rank", type=int, default=0, help='rank of machines')
    # 分布式训练使用的0号机器的ip，单机多卡训练设置为默认本机ip
    parser.add_argument("--master_addr", type=str, default="172.26.10.162",
                        help='ip address of machine 0')
    args = parser.parse_args()
    #################################
    # train(args.local_rank, args)：一般情况下保持local_rank与进程所用GPU_ID一致。
    print("----------")
    print(args.local_rank)
    print(args.batch_size)
    print("------------")
    return args


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    # with open('text/acc.txt', 'w') as f:
    #     with open('text/log.txt', 'w') as f2:
    #         for epoch in range(max_epoch):
    #             each_dist_tran_data_num = ((len(train_set)%dist.get_world_size()) + len(train_set)) /dist.get_world_size()
    #             train_sample.set_epoch(epoch)
    #
    #
    #             train_bar = tqdm(train_loader)
    #             optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    #             # ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    #             CosLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    #             model.train()
    #             train_loss = []
    #             train_accs = []
    #             for i, data in enumerate(train_bar):
    #                 inputs, labels = data
    #                 inputs, labels = inputs.to(device), labels.to(device)
    #                 outputs = model(inputs)
    #                 loss = criterion(outputs, labels)
    #                 optimizer.zero_grad()  # 每个batch的梯度需要清零
    #                 loss.backward()
    #                 optimizer.step()
    #                 acc = (outputs.argmax(dim=-1) == labels.to(device)).float().mean()
    #                 train_loss.append(loss.item())
    #                 train_accs.append(acc)
    #
    #                 train_bar.desc = "train epoch [{} / {}] loss:{:.3f} | acc:{:.3f}".format(epoch + 1, max_epoch, loss, 100. * sum(train_accs)/len(train_accs))
    #                 if i % 20 == 0:
    #                     # print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
    #                     #       % (epoch + 1, (i + 1 + epoch * len(train_loader)), sum(train_loss) / len(train_loss),
    #                     #          100. * sum(train_accs) / len(train_accs)))
    #                     f2.write(
    #                         '[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
    #                         % (epoch + 1, (i + 1 + epoch * len(train_loader)), sum(train_loss) / len(train_loss),
    #                            100. * sum(train_accs) / len(train_accs))
    #                     )
    #                     f2.write('\n')
    #                     f2.flush()
    #             #print("**************************Waiting Test!****************************", end='\n')
    #
    #             model.eval()
    #             valid_loss = []
    #             valid_accs = []
    #             with torch.no_grad():
    #                 test_bar = tqdm(test_loader)
    #                 for data in test_bar:
    #                     images, labels = data
    #                     images, labels = images.to(device), labels.to(device)
    #                     outputs = model(images)
    #
    #                     loss = criterion(outputs, labels.to(device))
    #                     acc = (outputs.argmax(dim=-1) == labels.to(device)).float().mean()
    #                     valid_loss.append(loss.item())
    #                     valid_accs.append(acc)
    #                     test_bar.desc = "valid epoch [{} / {}]".format(epoch+1, max_epoch)
    #             valid_loss = sum(valid_loss) / len(valid_loss)
    #             valid_acc = sum(valid_accs) / len(valid_accs)
    #             #print(f"[ Valid | {epoch + 1:03d}/{max_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    #
    #             #torch.save(model, 'model/net_%03d.pth' % (epoch + 1))
    #             f.write("epoch=%03d,Accuracy= %.3f%%" % (epoch + 1, valid_acc))
    #             f.write('\n')
    #             f.flush()
    #
    #             if valid_acc > best_acc:
    #                 f3 = open("text/best_acc.txt", "w")
    #                 f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, valid_acc))
    #                 f3.close()
    #                 best_acc = valid_acc
    args = parse_args()
    train(args.local_rank, args)
