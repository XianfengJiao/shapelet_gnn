import torch
from torch import nn
import os.path as op
import numpy as np
from tensorboardX import SummaryWriter
import torch.utils.data as data
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import functional as F

class GRU(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=64, output_dim=1, demo_dim=4, keep_prob=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.keep_prob = keep_prob
        self.demo_dim = demo_dim
        self.gru = torch.nn.GRU(self.input_dim, self.hidden_dim, batch_first = True)
        self.l_out = torch.nn.Linear(self.hidden_dim, self.output_dim)
        self.l_demo_out = torch.nn.Linear(self.hidden_dim * 2, self.output_dim)
        self.demo_proj = torch.nn.Linear(self.demo_dim, self.hidden_dim)
        self.tanh=torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=1 - keep_prob)
        
    def forward(self, x, lens, demo=None):
        batch_size = x.size(0)
        time_step = x.size(1)
        feature_dim = x.size(2)
        x = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden_t = self.gru(x)
        hn = hidden_t.squeeze(0)
        if self.keep_prob < 1.0:
            hn = self.dropout(hn)

        if demo != None:
            demo_embed = self.tanh(self.demo_proj(demo))# b hidden_dim
            o = self.l_demo_out(torch.cat([hn, demo_embed], 1))
        else:
            o = self.l_out(hn)

        
        o = self.sigmoid(o)

        return {'output': o.squeeze(-1)}