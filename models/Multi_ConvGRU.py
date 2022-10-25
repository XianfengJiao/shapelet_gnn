import torch
import sys
import pytorch_lightning as pl
import os.path as op
from torch import nn
import numpy as np
import torch.utils.data as data
from einops import rearrange, repeat
sys.path.append("/home/jxf/code/Shapelet_GNN")
from utils.model_utils import *


class ConvGRU(nn.Module):
    def __init__(
        self,
        feature_embed_dim=32, 
        num_channels=[4,8,16,32],
        kernel_size=4,
        keep_prob=1,
    ):
        super().__init__()
        # hyperparameters
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout_rate = 1-keep_prob
        self.feature_embed_dim = feature_embed_dim
        
        # layers
        self.feature_embed = nn.Linear(1, self.feature_embed_dim)
        conv_1ds = []
        grus = []
        for i in range(len(self.num_channels)):
            in_channels = self.feature_embed_dim if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            conv_1d = TemporalBlock(in_channels, out_channels, self.kernel_size, stride=1, padding=(self.kernel_size-1), dropout=self.dropout_rate, dilation=1)
            conv_1ds.append(conv_1d)
            gru = nn.GRU(out_channels, out_channels, batch_first = True)
            grus.append(gru)
        self.grus = nn.ModuleList(grus)
        self.conv_1ds = nn.ModuleList(conv_1ds)
    
    def forward(self, input, lens):
        if len(input.shape) == 2:
            input = input.unsqueeze(-1)
        batch_size = input.size(0)
        input = self.feature_embed(input)
        gru_embeded_output = []
        for i in range(len(self.num_channels)):
            conv_input = self.conv_1ds[i](input.permute(0,2,1)).transpose(1, 2)
            gru_input = pack_padded_sequence(conv_input, lens.cpu(), batch_first=True, enforce_sorted=False)
            _, gru_input = self.grus[i](gru_input)
            gru_embeded_output.append(gru_input.squeeze(0))
            input = conv_input
        
        gru_embeded_output = torch.cat(gru_embeded_output, dim=-1)
        return gru_embeded_output
        
class Multi_ConvGRU(nn.Module):
    def __init__(
        self, 
        input_dim=17, 
        output_dim=1,
        feature_embed_dim=32, 
        num_channels=[4,8,16,32], 
        kernel_size=4, 
        demo_dim=4, 
        keep_prob=1,
        demo_hidden_dim=32
    ):
        
        super().__init__()
        # hyperparameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.demo_dim = demo_dim
        self.demo_hidden_dim = demo_hidden_dim
        self.feature_embed_dim = feature_embed_dim
        self.dropout_rate = 1 - keep_prob
        
        # layers
        self.demo_proj_main = nn.Linear(self.demo_dim, self.demo_hidden_dim)
        conv_grus = [
            ConvGRU(
                feature_embed_dim=self.feature_embed_dim, 
                num_channels=self.num_channels, 
                kernel_size=self.kernel_size, 
                keep_prob=keep_prob
                )
            for _ in range(self.input_dim)
        ]
        self.conv_grus = nn.ModuleList(conv_grus)
        self.output = nn.Linear(sum(self.num_channels) * self.input_dim + self.demo_hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh=nn.Tanh()
        
    def forward(self, input, lens, demo_input):
        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)
        assert(feature_dim == self.input_dim) 
        
        demo_main = self.tanh(self.demo_proj_main(demo_input)).unsqueeze(1)# b hidden_dim
        
        embeded_inputs = []
        for i in range(feature_dim):
            conv_gru_o = self.conv_grus[i](input[:, :, i], lens)
            embeded_inputs.append(conv_gru_o)
        
        embeded_inputs = torch.cat(embeded_inputs, dim=-1)
        combined_hidden = torch.cat((embeded_inputs, \
                                     demo_main.squeeze(1)),-1)#b n h
        output = self.output(combined_hidden)
        output = self.sigmoid(output)
        
        return {'output': output.squeeze(-1)}