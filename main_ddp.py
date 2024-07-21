# -*- coding: utf-8 -*-

import os
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset


def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def setup():
    torch.distributed.init_process_group("nccl", init_method='env://')
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)


def cleanup():
    torch.distributed.destroy_process_group()


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


def get_dataset():
    x = torch.randn(1000, 10)
    y = torch.randn(1000, 10)
    dataset = TensorDataset(x, y)
    return dataset


def main(args):
    setup()
    torch.manual_seed(args.seed)

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # load data
    train_dataset = get_dataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank)
    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=int(args.num_workers),
        batch_size=int(args.batch_size),
        sampler=train_sampler)
    # val_loader / test_loader

    # for record
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # load model
    model = SimpleModel().to(device=rank)
    model = torch.nn.parallel.DistributedDataParallel(
        module=model,
        device_ids=[rank])

    # set criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=0.01,
        weight_decay=0.0005,
        momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=5,
        gamma=0.1)

    # train
    init_start_event.record()
    for epoch in range(1, 10 + 1):
        model.train()
        train_sampler.set_epoch(epoch)
        for data, target in train_loader:
            data, target = data.to(device=rank), target.to(device=rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        if rank == 0:
            print(f"Epoch {str(epoch).zfill(2)}, Loss: {round(loss.item(), 5)}")

        # val_loader / test_loader
        # model.eval()
        # with torch.no_grad():

        scheduler.step()
    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    # save model
    torch.distributed.barrier()
    states = model.state_dict()
    # if rank == 0:
    #     torch.save(states, "my_model.pt")

    cleanup()

    return


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    parser = argparse.ArgumentParser('DDP Training Test', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)


# torchrun --standalone --nnodes=1 --nproc_per_node=4 main_ddp.py
