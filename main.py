from backbone import resnet
from torchmetrics import Accuracy
from backbone.train_module import train_model
from torch.optim.lr_scheduler import StepLR
from data import dataProcess
import multiprocessing

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import SyncBatchNorm as SynBN


# def ddp_model(net, rank, world_size):
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)
#     net = net.to(rank)
#     ddp_net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
#     return ddp_net
#
# if __name__ == '__main__':
#     world_size, rank = 1, [2, 3]
#     mp.spawn(ddp_model, args=(world_size,), nprocs=world_size, join=True)
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '8703'
#     net = resnet.resnet34()
#     net = ddp_model(net, rank=rank, world_size=world_size)


import os
import argparse
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(object):
    def __init__(self, gpu: int, rank: int, world_size: int, model: nn.Module, dataset: DataLoader, loss_fn: nn.Module):
        self.gpu = gpu
        self.rank = rank
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.world_size = world_size


    def __call__(self, batch_size, lr, epochs=100, pin_memory=True):
        # load model
        device = torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() else 'cpu')
        model = self.model.to(device)
        if device.type != 'cpu':
            model = model if self.world_size == 1 else SynBN.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        # load dataset
        train_sampler =\
            torch.utils.data.distributed.DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.rank)
        train_loader = self.dataset

        # criterion = self.loss_fn.to(device)

        # create optimizer
        # optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        metrics_dict = {"acc": Accuracy(task='multiclass', num_classes=1000)}
        dfhistory = train_model(net=model, optimizer=None, loss_fn=self, metrics_dict=metrics_dict,
                                train_data=train_loader, val_data=None, epochs=epochs, lr=0.001,
                                patience=5, monitor="val_acc", mode="max", device=device)


def train(gpu, args):
    print(f'gpu:{gpu}')
    rank = args.nr * args.gpus + gpu
    # init process gropu
    dist.init_process_group(
        backend='nccl',
        world_size=args.world_size,
        rank=rank
    )

    # 声明训练器
    trainer = Trainer(
        gpu=gpu,
        rank=rank,
        world_size=args.world_size,
        model=args.model,
        dataset=args.dataset,
        loss_fn=nn.BCEWithLogitsLoss()
    )

    # start train
    trainer(batch_size=args.batch, lr=0.001, epochs=args.epochs, pin_memory=False)


if __name__ == '__main__':
    # set environment

    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8703'
    multiprocessing.set_start_method('spawn')
    os.chdir(os.path.join('/data', 'sft_lab', 'xjl', 'sync', 'icdar2024_sync_project'))
    os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

    # create dataset
    (dataloader_train, vocab_train), (dataloader_val, vocab_val) \
        = dataProcess.get_dataiter(2, None, is_train=True, num_images=10), (None, None)

    # create model
    model = resnet.resnet18()

    # parse argument
    parse = argparse.ArgumentParser('distribution setting')
    # 节点数/主机数
    parse.add_argument('-n', '--nodes', default=1, type=int, help='the number of nodes/computer')
    # 一个节点/主机上面的GPU数
    parse.add_argument('-g', '--gpus', default=1, type=int, help='the number of gpus per nodes')
    # 当前主机的编号，例如对于n机m卡训练，则nr∈[0,n-1]。对于单机多卡，nr只需为0。
    parse.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parse.add_argument('--epochs', default=10, type=int, help='number of epochs')
    args = parse.parse_args()
    args.batch = 2
    # 计算进程总数。进程总数实际上就是GPU总数
    args.world_size = args.nodes * args.gpus
    args.dataset = dataloader_train
    args.model = model

    mp.spawn(
        train,
        nprocs=args.gpus,
        args=(args,)
    )




#
# trans = torchvision.transforms.ToTensor()
# is_train = True
# is_location=True
# os.chdir(os.path.join('/data', 'sft_lab', 'xjl', 'sync', 'icdar2024_sync_project'))
# (dataloader_train, vocab_train), (dataloader_val, vocab_val) \
#     = dataProcess.get_dataiter(10, None, is_train=True, num_images=1000), (None, None)
# torch.manual_seed(3407)
# loss_fn = nn.BCEWithLogitsLoss()
# optimizer= torch.optim.Adam(net.parameters(), lr = 0.01)
# # scheduler = StepLR(optimizer=optimizer, step_size=50, gamma=0.3)
#
#
# print(len(vocab_train))
#
# metrics_dict = {"acc": Accuracy(task='multiclass', num_classes=len(vocab_train))}
#

