"""
Copyright (c) 2023 Yukara Ikemiya. All Rights Reserved.
Authored by Yukara Ikemiya.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, dim:int, num_data:int=10000):
        super().__init__()
        self.dim = dim
        self.num_data = num_data
    
    def get_item(self, idx):
        data = np.linspace(0, idx, self.dim)
        return torch.from_numpy(data.astype(np.float32))

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        return self.get_item(idx)
    
