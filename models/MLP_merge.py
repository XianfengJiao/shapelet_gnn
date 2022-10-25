import torch
from torch import nn
import os.path as op
import numpy as np
from tensorboardX import SummaryWriter
import torch.utils.data as data
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils.common_utils import clones


class MC_merge(nn.Module):
    def __init__(self, shapelet_hidden_size, labtest_hidden_size, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(shapelet_hidden_size+labtest_hidden_size, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.relu=nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, shapelet_embedding, labtest_embedding):
        h = self.fc1(torch.cat((shapelet_embedding, labtest_embedding), dim=-1))
        h = self.relu(h)
        o = self.fc_out(h)
        o = self.sigmoid(o)
        
        return o
        