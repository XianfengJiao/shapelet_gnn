import os
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.data import InMemoryDataset, download_url, extract_gz
import scipy.sparse as sp

class ShapeletDataset(InMemoryDataset):
    def __init__(self):
        super().__init__()
        
    @property
    def processed_file_names(self):
        return '/home/jxf/code/Shapelet_GNN/datasets/data.pt'
    
    def download(self):
        # Download to `self.raw_dir`.
        pass
        
    def process(self):
        # 创建节点属性
        data = HeteroData()
        
        
        
        # 创建边的邻接信息
        
        
        # 创建边的权值信息
        
        
        
        
        pass
    
    