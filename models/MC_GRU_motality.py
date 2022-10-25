import torch
from torch import nn
import pytorch_lightning as pl
import os.path as op
import numpy as np
import torch.utils.data as data
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils.common_utils import clones
# import sys
# sys.path.append("..")
# from common import *



class MC_GRU_motality(pl.LightningModule):
    def __init__(self, input_dim=17, demo_dim=4, hidden_dim=32, d_model=32,  MHD_num_head=4, d_ff=64, output_dim=1, keep_prob=0.5):
        super().__init__()

        # hyperparameters
        self.input_dim = input_dim  
        self.hidden_dim = hidden_dim  # d_model
        self.d_model = d_model
        self.MHD_num_head = MHD_num_head
 
        self.d_ff = d_ff
        self.output_dim = output_dim
        self.keep_prob = keep_prob
        self.demo_dim = demo_dim

        # layers
        self.GRUs = clones(nn.GRU(1, self.hidden_dim, batch_first = True), self.input_dim)

        self.dim_squeeze = nn.Linear(self.hidden_dim*(self.input_dim+1), self.hidden_dim)
        self.demo_proj = nn.Linear(self.demo_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.output_dim)

        self.dropout = nn.Dropout(p = 1 - self.keep_prob)
        self.tanh=nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()

    def forward(self, input, lens, demo_input):
        demo_main = self.tanh(self.demo_proj(demo_input)).unsqueeze(1)# b hidden_dim
        
        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)
        assert(feature_dim == self.input_dim)# input Tensor : 256 * 48 * 76
        assert(self.d_model % self.MHD_num_head == 0)

        
        GRU_embeded_input = self.GRUs[0](pack_padded_sequence(input[:,:,0].unsqueeze(-1), lens.cpu(), batch_first=True, enforce_sorted=False))[1].squeeze(0).unsqueeze(1) # b 1 h
        for i in range(feature_dim-1):
            embeded_input = self.GRUs[i+1](pack_padded_sequence(input[:,:,i+1].unsqueeze(-1), lens.cpu(), batch_first=True, enforce_sorted=False))[1].squeeze(0).unsqueeze(1) # b 1 h
            GRU_embeded_input = torch.cat((GRU_embeded_input, embeded_input), 1)

        GRU_embeded_input = torch.cat((GRU_embeded_input, demo_main), 1)# b i+1 h
        posi_input = self.dropout(GRU_embeded_input).view(batch_size, ((feature_dim+1) * self.hidden_dim)) # batch_size * d_input * hidden_dim
        
        posi_input = self.relu(self.dim_squeeze(posi_input))

        output = self.output(self.dropout(posi_input))# b 1
        output = self.sigmoid(output)
        
        return {'output': output.squeeze(-1)}
    