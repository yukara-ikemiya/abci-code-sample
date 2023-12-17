"""
Copyright (c) 2023 Yukara Ikemiya. All Rights Reserved.
Authored by Yukara Ikemiya.
"""

import argparse

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from simple_module import SimpleModule
from dummy_dataset import DummyDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=50, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--amp", type=str, default='fp16', help="autmatic mixed precision")
    return parser.parse_args()

def main():
    args = get_args()

    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=args.amp, split_batches=True)

    # Model
    model = SimpleModule(dim_in=1000, dim_hidden=100)

    # Dataset
    num_data = 10000
    dataset = DummyDataset(model.dim_in, num_data)
    dataloader = DataLoader(dataset, batch_size=args.bs, num_workers=4,
                            pin_memory=True, persistent_workers=True, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=[0.0, 0.99])

    # Prepare for distributed training
    model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)

    for idx_e in range(200):
        loss_epoch = 0.
        for idx_d, x in enumerate(dataloader):
            # forward
            optimizer.zero_grad(set_to_none=True)
            y = model(x)

            # RMSE loss
            loss = ((x - y) ** 2).mean().sqrt()

            # backward
            accelerator.backward(loss)
            optimizer.step()

            loss_epoch += loss.detach()

        if accelerator.is_main_process:
            loss_epoch /= idx_d + 1
            print(f'Epoch {idx_e+1} : {loss_epoch}')

if __name__ == '__main__':
    main()