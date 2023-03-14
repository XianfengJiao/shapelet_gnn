import numpy as np
import argparse
import os
import imp
import re
import pickle
import datetime
import random
import math
import sys
import logging
import copy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import kneighbors_graph

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.distributions.bernoulli as bernoulli

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
sys.path.append("/home/jxf/code/Shapelet_GNN")
from utils.model_utils import *


class SingleAttention(nn.Module):
    def __init__(self, attention_input_dim, attention_hidden_dim, attention_type='add', demographic_dim=12, time_aware=False, use_demographic=False):
        super(SingleAttention, self).__init__()
        
        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim
        self.use_demographic = use_demographic
        self.demographic_dim = demographic_dim
        self.time_aware = time_aware

        # batch_time = torch.arange(0, batch_mask.size()[1], dtype=torch.float32).reshape(1, batch_mask.size()[1], 1)
        # batch_time = batch_time.repeat(batch_mask.size()[0], 1, 1)
        
        if attention_type == 'add':
            if self.time_aware == True:
                # self.Wx = nn.Parameter(torch.randn(attention_input_dim+1, attention_hidden_dim))
                self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
                self.Wtime_aware = nn.Parameter(torch.randn(1, attention_hidden_dim))
                nn.init.kaiming_uniform_(self.Wtime_aware, a=math.sqrt(5))
            else:
                self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wt = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wd = nn.Parameter(torch.randn(demographic_dim, attention_hidden_dim))
            self.bh = nn.Parameter(torch.zeros(attention_hidden_dim,))
            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(torch.zeros(1,))
            
            nn.init.kaiming_uniform_(self.Wd, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == 'mul':
            self.Wa = nn.Parameter(torch.randn(attention_input_dim, attention_input_dim))
            self.ba = nn.Parameter(torch.zeros(1,))
            
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == 'concat':
            if self.time_aware == True:
                self.Wh = nn.Parameter(torch.randn(2*attention_input_dim+1, attention_hidden_dim))
            else:
                self.Wh = nn.Parameter(torch.randn(2*attention_input_dim, attention_hidden_dim))

            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(torch.zeros(1,))
            
            nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
            
        elif attention_type == 'new':
            self.Wt = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))

            self.rate = nn.Parameter(torch.zeros(1)+0.8)
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
            
        else:
            raise RuntimeError('Wrong attention type.')
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, input, demo=None):
 
        batch_size, time_step, input_dim = input.size() # batch_size * time_step * hidden_dim(i)
        #assert(input_dim == self.input_dim)

        # time_decays = torch.zeros((time_step,time_step)).to(device)# t*t
        # for this_time in range(time_step):
        #     for pre_time in range(time_step):
        #         if pre_time > this_time:
        #             break
        #         time_decays[this_time][pre_time] = torch.tensor(this_time - pre_time, dtype=torch.float32).to(device)
        # b_time_decays = tile(time_decays, 0, batch_size).view(batch_size,time_step,time_step).unsqueeze(-1).to(device)# b t t 1

        time_decays = torch.tensor(range(47,-1,-1), dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(input.device)# 1*t*1
        b_time_decays = time_decays.repeat(batch_size,1,1)+1# b t 1
        
        if self.attention_type == 'add': #B*T*I  @ H*I
            q = torch.matmul(input[:,-1,:], self.Wt)# b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim)) #B*1*H
            if self.time_aware == True:
                # k_input = torch.cat((input, time), dim=-1)
                k = torch.matmul(input, self.Wx)#b t h
                # k = torch.reshape(k, (batch_size, 1, time_step, self.attention_hidden_dim)) #B*1*T*H
                time_hidden = torch.matmul(b_time_decays, self.Wtime_aware)#  b t h
            else:
                k = torch.matmul(input, self.Wx)# b t h
                # k = torch.reshape(k, (batch_size, 1, time_step, self.attention_hidden_dim)) #B*1*T*H
            if self.use_demographic == True:
                d = torch.matmul(demo, self.Wd) #B*H
                d = torch.reshape(d, (batch_size, 1, self.attention_hidden_dim)) # b 1 h
            h = q + k + self.bh # b t h
            if self.time_aware == True:
                h += time_hidden
            h = self.tanh(h) #B*T*H
            e = torch.matmul(h, self.Wa) + self.ba #B*T*1
            e = torch.reshape(e, (batch_size, time_step))# b t
        elif self.attention_type == 'mul':
            e = torch.matmul(input[:,-1,:], self.Wa)#b i
            e = torch.matmul(e.unsqueeze(1), input.permute(0,2,1)).squeeze() + self.ba #b t
        elif self.attention_type == 'concat':
            q = input[:,-1,:].unsqueeze(1).repeat(1,time_step,1)# b t i
            k = input
            c = torch.cat((q, k), dim=-1) #B*T*2I
            if self.time_aware == True:
                c = torch.cat((c, b_time_decays), dim=-1) #B*T*2I+1
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba #B*T*1
            e = torch.reshape(e, (batch_size, time_step)) # b t 
            
        elif self.attention_type == 'new':
            
            q = torch.matmul(input[:,-1,:], self.Wt)# b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim)) #B*1*H
            k = torch.matmul(input, self.Wx)#b t h
            dot_product = torch.matmul(q, k.transpose(1, 2)).squeeze() # b t
            denominator =  self.sigmoid(self.rate) * (torch.log(2.72 +  (1-self.sigmoid(dot_product)))* (b_time_decays.squeeze()))
            e = self.relu(self.sigmoid(dot_product)/(denominator)) # b * t
#          * (b_time_decays.squeeze())
        # e = torch.exp(e - torch.max(e, dim=-1, keepdim=True).values)
        
        # if self.attention_width is not None:
        #     if self.history_only:
        #         lower = torch.arange(0, time_step).to(device) - (self.attention_width - 1)
        #     else:
        #         lower = torch.arange(0, time_step).to(device) - self.attention_width // 2
        #     lower = lower.unsqueeze(-1)
        #     upper = lower + self.attention_width
        #     indices = torch.arange(0, time_step).unsqueeze(0).to(device)
        #     e = e * (lower <= indices).float() * (indices < upper).float()
        
        # s = torch.sum(e, dim=-1, keepdim=True)
        # mask = subsequent_mask(time_step).to(device) # 1 t t 下三角
        # scores = e.masked_fill(mask == 0, -1e9)# b t t 下三角
        a = self.softmax(e) #B*T
        v = torch.matmul(a.unsqueeze(1), input).squeeze() #B*I

        return v, a

class FinalAttentionQKV(nn.Module):
    def __init__(self, attention_input_dim, attention_hidden_dim, attention_type='add', dropout=None):
        super(FinalAttentionQKV, self).__init__()
        
        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim


        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)

        self.W_out = nn.Linear(attention_hidden_dim, 1)

        self.b_in = nn.Parameter(torch.zeros(1,))
        self.b_out = nn.Parameter(torch.zeros(1,))

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.Wh = nn.Parameter(torch.randn(2*attention_input_dim, attention_hidden_dim))
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(torch.zeros(1,))
        
        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        
        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
 
        batch_size, time_step, input_dim = input.size() # batch_size * input_dim + 1 * hidden_dim(i)
        input_q = self.W_q(input[:, -1, :]) # b h
        input_k = self.W_k(input)# b t h
        input_v = self.W_v(input)# b t h

        if self.attention_type == 'add': #B*T*I  @ H*I

            q = torch.reshape(input_q, (batch_size, 1, self.attention_hidden_dim)) #B*1*H
            h = q + input_k + self.b_in # b t h
            h = self.tanh(h) #B*T*H
            e = self.W_out(h) # b t 1
            e = torch.reshape(e, (batch_size, time_step))# b t

        elif self.attention_type == 'mul':
            q = torch.reshape(input_q, (batch_size, self.attention_hidden_dim, 1)) #B*h 1
            e = torch.matmul(input_k, q).squeeze()#b t
            
        elif self.attention_type == 'concat':
            q = input_q.unsqueeze(1).repeat(1,time_step,1)# b t h
            k = input_k
            c = torch.cat((q, k), dim=-1) #B*T*2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba #B*T*1
            e = torch.reshape(e, (batch_size, time_step)) # b t 
        
        a = self.softmax(e) #B*T
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze() #B*I

        return v, a

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
    return torch.index_select(a, dim, order_index).to(a.device)

class PositionwiseFeedForward(nn.Module): # new added
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x)))), None

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0 # 下三角矩阵

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)# b h t d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k) # b h t t
    if mask is not None:# 1 1 t t
        scores = scores.masked_fill(mask == 0, -1e9)# b h t t 下三角
    p_attn = F.softmax(scores, dim = -1)# b h t t
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn # b h t v (d_k) 
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, self.d_k * self.h), 3)
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1) # 1 1 t t

        nbatches = query.size(0)# b
        input_dim = query.size(1)# i+1
        feature_dim = query.size(-1)# i+1

        #input size -> # batch_size * d_input * hidden_dim
        
        # d_model => h * d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))] # b num_head d_input d_k
        
       
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)# b num_head d_input d_v (d_k) 

      
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)# batch_size * d_input * hidden_dim

        #DeCov 
        DeCov_contexts = x.transpose(0, 1).transpose(1, 2) # I+1 H B
        Covs = cov(DeCov_contexts[0,:,:])
        DeCov_loss = 0.5 * (torch.norm(Covs, p = 'fro')**2 - torch.norm(torch.diag(Covs))**2 ) 
        for i in range(input_dim - 1):
            Covs = cov(DeCov_contexts[i+1,:,:])
            DeCov_loss += 0.5 * (torch.norm(Covs, p = 'fro')**2 - torch.norm(torch.diag(Covs))**2 ) 


        return self.final_linear(x), DeCov_loss

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        returned_value = sublayer(self.norm(x))
        return x + self.dropout(returned_value[0]) , returned_value[1]

class ConvGRU_attn(nn.Module):
    def __init__(
        self,
        feature_embed_dim=32, 
        num_channels=[4,8,16,32],
        kernel_size=4,
        attention_hidden_dim=32,
        keep_prob=1,
    ):
        super().__init__()
        # hyperparameters
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout_rate = 1-keep_prob
        self.feature_embed_dim = feature_embed_dim
        self.attention_hidden_dim = attention_hidden_dim
        
        # layers
        self.feature_embed = nn.Linear(1, self.feature_embed_dim)
        self.pos_encoder = PositionalEncoding(self.feature_embed_dim)
        conv_1ds = []
        grus = []
        attentions = []
        for i in range(len(self.num_channels)):
            in_channels = self.feature_embed_dim if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            conv_1d = TemporalBlock(in_channels, out_channels, self.kernel_size, stride=1, padding=(self.kernel_size-1), dropout=self.dropout_rate, dilation=1)
            conv_1ds.append(conv_1d)
            attention = Attention(attention_input_dim = out_channels, 
                                  attention_hidden_dim = self.attention_hidden_dim, 
                                  output_dim=out_channels, 
                                  feature_dim = out_channels, 
                                  attention_type = 'mul', 
                                  dropout = self.dropout_rate,
                                  attention_act='softmax')
            attentions.append(attention)
            gru = nn.GRU(out_channels, out_channels, batch_first = True)
            grus.append(gru)
            
        self.grus = nn.ModuleList(grus)
        self.conv_1ds = nn.ModuleList(conv_1ds)
        self.attentions = nn.ModuleList(attentions)
        
    def gen_key_padding_mask(self, shape, lens):
        # Considering the cls_token
        shape = list(shape)
        mask = torch.full(shape, False)
        for i, l in enumerate(lens):
            mask[i][l:-1] = True
        return mask 
    
    def forward(self, input, lens):
        if len(input.shape) == 2:
            input = input.unsqueeze(-1)
        batch_size = input.size(0)
        key_padding_mask = self.gen_key_padding_mask(input.shape[:2], lens).type_as(input).type(torch.bool)
        input = self.feature_embed(input)
        embeded_output = []
        attns = []
        for i in range(len(self.num_channels)):
            conv_input = self.conv_1ds[i](input.permute(0,2,1)).transpose(1, 2)
            gru_input = pack_padded_sequence(conv_input, lens.cpu(), batch_first=True, enforce_sorted=False)
            _, gru_query = self.grus[i](gru_input)
            attn_input, attn = self.attentions[i](query=gru_query.squeeze(0), input=conv_input, src_key_padding_mask=key_padding_mask)
            embeded_output.append(attn_input)
            attns.append(attn.unsqueeze(1))
            input = conv_input
        
        embeded_output = torch.cat(embeded_output, dim=-1)
        attns = torch.cat(attns, dim=1)
        return embeded_output, attns


class ConCare_mimiic_conv(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_model=64, num_channels=[4,8,16,32], kernel_size=4,  MHD_num_head=4, demo_dim=12, d_ff=256, output_dim=1, keep_prob=0.5):
        super(ConCare_mimiic_conv, self).__init__()

        # hyperparameters
        self.input_dim = input_dim  
        self.hidden_dim = hidden_dim  # d_model
        self.d_model = d_model
        self.MHD_num_head = MHD_num_head
        self.d_ff = d_ff
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.keep_prob = keep_prob
        self.demo_dim = demo_dim

        # layers
        # self.PositionalEncoding = PositionalEncoding(self.d_model, dropout = 0, max_len = 400)
        # self.GRUs = clones(nn.GRU(1, self.hidden_dim, batch_first = True), self.input_dim)
        
        conv_grus = [
            ConvGRU_attn(
                feature_embed_dim=self.hidden_dim, 
                num_channels=self.num_channels, 
                kernel_size=self.kernel_size, 
                keep_prob=keep_prob
                )
            for _ in range(self.input_dim)
        ]
        self.conv_grus = nn.ModuleList(conv_grus)
        self.projector = nn.Linear(sum(self.num_channels), self.hidden_dim)
        
        self.LastStepAttentions = clones(SingleAttention(self.hidden_dim, 8, attention_type='new', demographic_dim=demo_dim, time_aware=True, use_demographic=False),self.input_dim)
        
        self.FinalAttentionQKV = FinalAttentionQKV(self.hidden_dim, self.hidden_dim, attention_type='mul',dropout = 1 - self.keep_prob)

        self.MultiHeadedAttention = MultiHeadedAttention(self.MHD_num_head, self.hidden_dim, dropout = 1 - self.keep_prob)
        self.SublayerConnection = SublayerConnection(self.hidden_dim, dropout = 1 - self.keep_prob)

        self.PositionwiseFeedForward = PositionwiseFeedForward(self.hidden_dim, self.d_ff, dropout=0.1)

        self.demo_proj_main = nn.Linear(demo_dim, self.hidden_dim)
        self.demo_proj = nn.Linear(demo_dim, self.hidden_dim)
        self.output0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output1 = nn.Linear(self.hidden_dim, self.output_dim)

        self.dropout = nn.Dropout(p = 1 - self.keep_prob)
        self.tanh=nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu=nn.ReLU()

    def forward(self, input, lens, demo_input):
        # input shape [batch_size, timestep, feature_dim]
        demo_main = self.tanh(self.demo_proj_main(demo_input)).unsqueeze(1)# b 1 hidden_dim
        
        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)
        assert(feature_dim == self.input_dim)# input Tensor : 256 * 48 * 76
        assert(self.d_model % self.MHD_num_head == 0)

        # Initialization
        #cur_hs = Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0))

        # forward
        embeded_inputs = []
        attns = []
        for i in range(feature_dim):
            conv_gru_o, attn = self.conv_grus[i](input[:, :, i], lens)
            embeded_inputs.append(conv_gru_o.unsqueeze(1))
            attns.append(attn.unsqueeze(1))
        
        attns = torch.cat(attns, dim=1)
        embeded_inputs = torch.cat(embeded_inputs, dim=1)
        embeded_inputs = self.projector(embeded_inputs)
        embeded_inputs = torch.cat((embeded_inputs, demo_main), 1)
        embeded_inputs = self.dropout(embeded_inputs) # batch_size * d_input+1 * hidden_dim

        #mask = subsequent_mask(time_step).to(device) # 1 t t 下三角 N to 1任务不用mask
        contexts = self.SublayerConnection(embeded_inputs, lambda x: self.MultiHeadedAttention(embeded_inputs, embeded_inputs, embeded_inputs, None))# # batch_size * d_input * hidden_dim
    
        DeCov_loss = contexts[1]
        contexts = contexts[0]

        contexts = self.SublayerConnection(contexts, lambda x: self.PositionwiseFeedForward(contexts))[0]# # batch_size * d_input * hidden_dim

        weighted_contexts = self.FinalAttentionQKV(contexts)[0]
        output = self.output1(self.relu(self.output0(weighted_contexts)))# b 1
        output = self.sigmoid(output)
          
        return {'output': output.squeeze(-1), 'decov_loss': DeCov_loss, 'emb': weighted_contexts}

