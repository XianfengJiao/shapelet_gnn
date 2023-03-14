import os
from re import X
import sys
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import gc
from torch.utils.data import Dataset

class PatientDataset(Dataset):
    def __init__(
        self,
        x_data,
        y_data,
        static_data,
        pdid_data,
        lens_data=None
        ):
        self.x = x_data
        self.y = y_data
        self.pdid = pdid_data
        self.static = static_data
        if type(lens_data) == type(None):
            self.lens = [len(i) for i in self.x]
        else:
            self.lens = lens_data
        

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        static = self.static[index]
        lens = self.lens[index]
        pdid = self.pdid[index]
        return x, y, static, lens, pdid
    
    @staticmethod
    def collate_fn(dataset):
        x, y, static, lens, pdid = zip(*dataset)
        check_lens = [len(xx) for xx in x]
        if max(check_lens) == min(check_lens):
            return torch.FloatTensor(x), torch.FloatTensor(y), torch.FloatTensor(static), torch.LongTensor(lens), torch.LongTensor(pdid)
        if len(np.array(x[0]).shape) == 1:
            x_pad = torch.zeros(len(x), max(lens), 1).float()
        else: 
            x_pad = torch.zeros(len(x), max(lens), len(x[0][0])).float()
        for i, xx in enumerate(x):
            end = lens[i]
            x_pad[i,:end] = torch.FloatTensor(np.array(xx, dtype=float)).unsqueeze(1) if len(np.array(xx, dtype=float).shape) == 1 else torch.FloatTensor(np.array(xx, dtype=float))
        return x_pad, torch.FloatTensor(y), torch.FloatTensor(static), torch.LongTensor(lens), torch.LongTensor(pdid)