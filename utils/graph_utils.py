import os
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero
from torch.nn import functional as F
from argparse import ArgumentParser
import pickle as pkl
import numpy as np
from glob import glob
import random
import torch
import scipy.sparse as sp
from torch_geometric.data import Data, HeteroData

def build_graph(graph_path, node_dim=8):
    data = HeteroData()
    
    evo_fps = glob(os.path.join(graph_path, '*tmat.pkl'))
    evo_g = {}
    # build evolution graph
    for evo_fp in evo_fps:
        i = os.path.basename(evo_fp).split('-')[1].split('_')[0]
        with open(evo_fp, 'rb') as f:
            evo_g['f'+i] = pkl.load(f)
        node_size = len(evo_g['f'+i])
        data['f'+i].x = torch.rand((node_size, node_dim))
        edge_index_temp = sp.coo_matrix(evo_g['f'+i])
        weights = edge_index_temp.data
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
        data['f'+i, 'evolve', 'f'+i].edge_index = torch.LongTensor(indices)
        data['f'+i, 'evolve', 'f'+i].edge_attr = torch.FloatTensor(weights).unsqueeze(-1)
    
    # build co-occurrence
    co_occu_g = pkl.load(open(os.path.join(graph_path, 'co_mat.pkl'), 'rb'))
    
    node_type_size = len(evo_g)
    node_size=len(evo_g['f0'])
    for i in range(node_type_size - 1):
        for j in range(i+1, node_type_size):
            select_co_occu_g = co_occu_g[i*50:(i+1)*50, j*50:(j+1)*50]
            edge_index_temp = sp.coo_matrix(select_co_occu_g)
            weights = edge_index_temp.data
            indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
            data['f'+str(i), 'co_occur', 'f'+str(j)].edge_index = torch.LongTensor(indices)
            data['f'+str(i), 'co_occur', 'f'+str(j)].edge_attr = torch.FloatTensor(weights).unsqueeze(-1)
            # TODO: check
            indices = np.vstack((edge_index_temp.col, edge_index_temp.row))
            data['f'+str(j), 'co_occur', 'f'+str(i)].edge_index = torch.LongTensor(indices)
            data['f'+str(j), 'co_occur', 'f'+str(i)].edge_attr = torch.FloatTensor(weights).unsqueeze(-1)
    
    return data
