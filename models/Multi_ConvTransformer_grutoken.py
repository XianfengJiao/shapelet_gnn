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

class PositionalEncoding(nn.Module): 
	def __init__(self, d_model, dropout=0, max_len=500):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) *
			-(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
		return self.dropout(x)

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class Attention(nn.Module):
    def __init__(self, attention_input_dim, attention_hidden_dim, output_dim,  feature_dim, attention_type='add', dropout=None, attention_act='softmax'):
        super(Attention, self).__init__()
        
        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim
        self.attention_act = attention_act
        self.feature_dim = feature_dim


        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.output = nn.Linear(attention_hidden_dim, output_dim)

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
        self.rate = nn.Parameter(torch.ones(self.feature_dim))
        
        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        
        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sparsemax = Sparsemax()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, query, input, time_mask=None, src_key_padding_mask=None):
        batch_size, time_step, input_dim = input.size() # batch_size * input_dim + 1 * hidden_dim(i)
        input_q = self.W_q(query) # b h
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
            e = torch.matmul(input_k, q).squeeze(-1)#b t
            if time_mask != None:
                time_miss = torch.log(1 + (1 - self.sigmoid(e)) * (time_mask.squeeze()))
                e = e - self.rate * time_miss

            
        elif self.attention_type == 'concat':
            q = input_q.unsqueeze(1).repeat(1,time_step,1)# b t h
            k = input_k
            c = torch.cat((q, k), dim=-1) #B*T*2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba #B*T*1
            e = torch.reshape(e, (batch_size, time_step)) # b t 
        
        if src_key_padding_mask is not None:
            e = e.masked_fill(src_key_padding_mask, value=torch.tensor(-1e9))
        
        if self.attention_act == 'sparsemax':
            a = self.sparsemax(e) #B*T
        else:
            a = self.softmax(e) #B*T
        
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze(1) #B*I
        o = self.output(v)
        return o, a


class ConvTransformer(pl.LightningModule):
    def __init__(self, feature_embed_dim=4, num_channels=[4,8,16,32], attention_hidden_dim=32, kernel_size=2, keep_prob=1, nhead=4, num_layers=3):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.keep_prob = keep_prob
        self.nhead = nhead
        self.num_layers = num_layers
        self.feature_embed_dim = feature_embed_dim
        self.attention_hidden_dim = attention_hidden_dim
        
        self.feature_embed = nn.Linear(1, self.feature_embed_dim)
        self.pos_encoder = PositionalEncoding(self.feature_embed_dim)
        
        cls_tokens = []
        attentions = []
        cross_attentions = []
        conv_1ds = []
        transformer_encoders = []
        cls_embeds = []
        grus = []
        
        
        dropout = 1-self.keep_prob
        for i in range(len(self.num_channels)):
            in_channels = self.feature_embed_dim if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            encoder_layer = nn.TransformerEncoderLayer(d_model=out_channels, nhead=self.nhead, dropout=dropout, batch_first=True)
            transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
            conv_1d = TemporalBlock(in_channels, out_channels, self.kernel_size, stride=1, padding=(self.kernel_size-1), dropout=dropout, dilation=1)
            cls_token = nn.Parameter(torch.zeros(1, 1, out_channels))
            attention = Attention(attention_input_dim = out_channels, 
                                  attention_hidden_dim = self.attention_hidden_dim, 
                                  output_dim=out_channels, 
                                  feature_dim = out_channels, 
                                  attention_type = 'mul', 
                                  dropout = 1 - self.keep_prob,
                                  attention_act='softmax')
            gru = nn.GRU(out_channels, out_channels, batch_first = True)
            grus.append(gru)
            attentions.append(attention)
            cls_tokens.append(cls_token)
            cls_embed = nn.Linear(out_channels, self.num_channels[-1])
            conv_1ds.append(conv_1d)
            transformer_encoders.append(transformer_encoder)
            cls_embeds.append(cls_embed)
        
        self.cls_tokens = nn.ParameterList(cls_tokens)
        self.conv_1ds = nn.ModuleList(conv_1ds)
        self.grus = nn.ModuleList(grus)
        self.attentions = nn.ModuleList(attentions)
        self.transformer_encoders = nn.ModuleList(transformer_encoders)
        self.cls_embeds = nn.ModuleList(cls_embeds)
        self.sigmoid = nn.Sigmoid()
    
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
        input = self.pos_encoder(input)
        cls_tokens = []
        attns = []
        for i in range(len(self.num_channels)):
            conv_input = self.conv_1ds[i](input.permute(0,2,1)).transpose(1, 2)
            # cls_token = repeat(self.cls_tokens[i], '() n d -> b n d', b=batch_size).type_as(input)
            # cls_conv_input = torch.cat((conv_input, cls_token), dim=1) 
            attn_input = self.transformer_encoders[i](conv_input, src_key_padding_mask=key_padding_mask)
            gru_input = pack_padded_sequence(attn_input, lens.cpu(), batch_first=True, enforce_sorted=False)
            _, gru_query = self.grus[i](gru_input)
            cls_token_attn, attn = self.attentions[i](query=gru_query.squeeze(0), input=attn_input, src_key_padding_mask=key_padding_mask)
            cls_tokens.append(self.cls_embeds[i](cls_token_attn))
            attns.append(attn.unsqueeze(1))
            input = attn_input
        
        cls_output = torch.cat(cls_tokens, dim=1)
        attns = torch.cat(attns, dim=1)
        
        return cls_output, attns

class Multi_ConvTransformer_grutoken(pl.LightningModule):
    def __init__(self, input_dim=17, output_dim=1, feature_embed_dim = 32, demo_dim = 4, demo_hidden_dim = 32, num_channels=[4,8,16,32], attention_hidden_dim=32, kernel_size=4, keep_prob=1, nhead=4, num_layers=3):
        super().__init__()
        # hyperparameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.keep_prob = keep_prob
        self.nhead = nhead
        self.num_layers = num_layers
        self.feature_embed_dim = feature_embed_dim
        self.attention_hidden_dim = attention_hidden_dim
        self.demo_dim = demo_dim
        self.demo_hidden_dim = demo_hidden_dim
        
        # layers
        self.demo_proj_main = nn.Linear(self.demo_dim, self.demo_hidden_dim)
        conv_transformers = [
            ConvTransformer(feature_embed_dim = self.feature_embed_dim, 
                            num_channels=self.num_channels, 
                            attention_hidden_dim=self.attention_hidden_dim, 
                            kernel_size=self.kernel_size, 
                            keep_prob=self.keep_prob, 
                            nhead=self.nhead, 
                            num_layers=self.num_layers) \
                for _ in range(self.input_dim)
            ]
        self.conv_transformers = nn.ModuleList(conv_transformers)
        self.FinalAttentionQKV = FinalAttentionQKV(self.num_channels[-1] * len(self.num_channels), self.num_channels[-1] * len(self.num_channels), self.input_dim, attention_type='mul',dropout = 1 - self.keep_prob)
        self.decoder = nn.Linear(self.num_channels[-1] * len(self.num_channels) + self.demo_hidden_dim, 1)
        self.init_weights()
        self.sigmoid = nn.Sigmoid()
        self.tanh=nn.Tanh()
    
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, input, lens, demo_input):
        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)
        assert(feature_dim == self.input_dim)        
        
        demo_main = self.tanh(self.demo_proj_main(demo_input)).unsqueeze(1)# b hidden_dim
        
        f_cls_outputs = []
        attns = []
        for i in range(feature_dim):
            f_cls_o, attn = self.conv_transformers[i](input[:, :, i], lens)
            f_cls_outputs.append(f_cls_o.unsqueeze(1))
            attns.append(attn.unsqueeze(1))
        
        attns = torch.cat(attns, dim=1)
        f_cls_outputs = torch.cat(f_cls_outputs, dim=1)
        
        weighted_contexts, f_attns = self.FinalAttentionQKV(f_cls_outputs)
        
        combined_hidden = torch.cat((weighted_contexts, \
                                     demo_main.squeeze(1)),-1)#b n h
        output = self.decoder(combined_hidden)
        output = self.sigmoid(output)

        return {'output': output.squeeze(-1), 'attns': attns, 'f_attns': f_attns}