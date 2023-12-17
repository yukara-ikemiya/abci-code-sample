"""
Copyright (c) 2023 Yukara Ikemiya. All Rights Reserved.
Authored by Yukara Ikemiya.
"""

import torch
from torch import nn

# simplest auto-encoder

class SimpleModule(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()

        self.dim_in = dim_in
        self.dim_hidden = dim_hidden

        self.net = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_hidden),
            nn.Linear(self.dim_hidden, self.dim_in)
        )

    def forward(self, x):
        return self.net(x)