import argparse
import torchvision
import torch.distributed as dist
import torch.utils.data.distributed
import torchvision.transforms as transforms

from tqdm import tqdm
from mobilenetv2 import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def evalutate(model, gpu, test_loader, rank, epoch):
    if rank == 1:
        return
    test_bar = tqdm(test_loader)
    val_accus = []
    model.eval()
    size = torch.tensor(0.).to(gpu)
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_bar):
            images = images.to(gpu)
            labels = labels.to(gpu)
            outputs = model(images)
            size += images.shape[0]
            acc = (outputs.argmax(dim=-1) == labels.to(gpu)).float().mean()
            val_accus.append(acc)
            test_bar.desc = "Val epoch [{} / {}], Acc:{:.3f}".format(epoch + 1, 50,100. * sum(val_accus) /size)

def train(gpu, args):
    # 训练函数中仅需要更改初始化方式即可。在ENV中只需要指定init_method='env://'。
    # TCP所需的关键参数模型会从环境变量中自动获取，环境变量可以在程序外部启动时设定，参考启动方式。
    dist.init_process_group(backend='nccl', init_method='env://', world_size=2)
    args.rank = dist.get_rank()

    model = MobileNetV2(num_classes=10)
    model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().to(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # 并行环境下，对于用到BN层的模型需要转换为同步BN层；
    # 用Distributed Dataparallel将模型封装为一个DDP模型，并复制到指定的GPU上
    # 封装时不需要改变模型内部的代码；设置混合精度中scaler，通过设置enabled参数控制是否生效
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids = [gpu])
    #scaler = GradScaler(enabled=args.use_mix_precision)

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
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)  # num_workers=0)

    total_step = len(train_loader)      #这里的意思是按batch_size的大小，总共有多少份

    for epoch in range(args.epochs):
        # 在每个epoch开始前打乱数据顺序
        train_loader.sampler.set_epoch(epoch)
        model.train()
        if args.rank ==0:
            train_bar = tqdm(train_loader)
        else:
            train_bar = train_loader
        train_accs = []
        for i,(images,labels) in enumerate(train_bar):
            images= images.to(gpu)
            labels = labels.to(gpu)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (outputs.argmax(dim=-1) == labels.to(gpu)).float().mean()
            train_accs.append(acc)
            # 避免LOG信息重复打印，只允许rank0进程打印
            if args.rank==0:
                train_bar.desc = "Train epoch [{} / {}], Step[{}/{}],  loss:{:.3f} | acc:{:.3f}".format(
                    epoch + 1, 50, i+1, total_step,loss.item(),100. * sum(train_accs) / len(train_accs))
        evalutate(model, gpu, test_loader, args.rank, epoch)
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpuid', default=0, type=int, help = "which gpu to use")
    parser.add_argument('-e', '--epochs', default=50, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=96, type=int,
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
    return args

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    args = parse_args()
    train(args.local_rank, args)
