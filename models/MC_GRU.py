import torch
from torch import nn
import os.path as op
import numpy as np
from tensorboardX import SummaryWriter
import torch.utils.data as data
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils.common_utils import clones

class MC_GRU(nn.Module):
    def __init__(self, feature_size ,input_dim, hidden_dim, output_dim, keep_prob=0.5):
        super().__init__()
        
        self.GRUs = clones(nn.GRU(input_dim, hidden_dim, batch_first = True), feature_size)
        self.dim_squeeze = nn.Linear(hidden_dim*feature_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(p = 1 - keep_prob)
        self.tanh=nn.Tanh()
    
    def forward(self, input, lens):
        feature_dim = input.size(1)
        GRU_embeded_input = torch.Tensor().to(input.device)
        for i in range(feature_dim):
            embeded_input = self.GRUs[i](
                pack_padded_sequence(input[:,i,:,:], lens, batch_first=True, enforce_sorted=False)
                )[1].squeeze(0) # b h
            GRU_embeded_input = torch.cat((GRU_embeded_input, embeded_input), 1)
        
        GRU_embeded_input = self.relu(self.dim_squeeze(self.dropout(GRU_embeded_input)))
        
        output = self.output(self.dropout(GRU_embeded_input))
        output = self.sigmoid(output).squeeze(1)
        
        return output
