import os
from re import X
import sys
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import gc
from torch.utils.data import Dataset


class RecordDataset(Dataset):
    def __init__(self, input_data, label_data, emb_data, data_dir, type, topk=5):
        preprocessed_path = os.path.join(data_dir, type+'_record_'+'preprocessed.pkl')
        os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
        
        input_data, label_data, emb_data = self.filter_data(input_data, label_data, emb_data)
        
        if not os.path.isfile(preprocessed_path):
            patient_size = len(input_data['f0'])
            feature_size = len(input_data)
            record_preprocessed = [[[] for _ in range(feature_size)] for _ in range(patient_size)]
            # feature_size x patient_size x record_size x shapelet_size
            for key, x in input_data.items():
                i = int(key.replace('f',''))
                for p_i, p_x in enumerate(x):
                    record_preprocessed[p_i][i] = self.process_data_per_patient(p_x, topk=topk)
            
            pkl.dump(record_preprocessed, open(preprocessed_path, 'wb'))
        else:
            # -------------------------- Load Preprocessed DATA --------------------------
            print("Found preprocessed record data. Loading that!")
            record_preprocessed = pkl.load(open(preprocessed_path, 'rb'))
            
        # return patient_size x feature_size x record_size x shapelet_size
        self.data = list(zip(record_preprocessed, label_data, emb_data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def filter_data(self, data, label, emb):
        label = [ll for (ll, dd) in zip(label, data['f0']) if len(dd) > 0]
        emb = [ee for (ee, dd) in zip(emb, data['f0']) if len(dd) > 0]
        for key in data.keys():
            data[key] = [dd for dd in data[key] if len(dd) > 0]
        
        return data, label, emb
    
    def process_data_per_patient(self, data, topk=5):
        # data: record_size x shapelet_size
        data = torch.tensor(data)
        tmp =  torch.rand_like(data.float())
        topk_value = torch.topk(data, topk, largest=False).values[:, -1]
        for i, v in enumerate(topk_value):
            tmp[i, :] = v
        return (data <= tmp).float()
    
    @staticmethod
    def collate_fn(dataset):
        x, y, e = zip(*dataset)
        lens = [len(d[0]) for d in x]
        x_pad = torch.zeros(len(x), len(x[0]), max(lens), len(x[0][0][0])).float()
        for p_i, f_x in enumerate(x):
            end = lens[p_i]
            for f_i, xx in enumerate(f_x):
                x_pad[p_i,f_i,:end] = torch.FloatTensor(xx)
        return x_pad, torch.FloatTensor(y), torch.LongTensor(lens), torch.FloatTensor(e)