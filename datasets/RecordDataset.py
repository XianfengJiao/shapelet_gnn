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
    def __init__(self, input_data, label_data, data_dir, type, fold, topk=5):
        preprocessed_path = os.path.join(data_dir, 'fold-'+str(fold), type+'_record_'+'preprocessed.pkl')
        os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
        
        if not os.path.isfile(preprocessed_path):
            patient_size = len(input_data['f0'])
            record_preprocessed = [[[] for _ in range(4)] for _ in range(patient_size)]
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
        self.data = list(zip(record_preprocessed, label_data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
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
        x, y = zip(*dataset)
        lens = [len(d[0]) for d in x]
        x_pad = torch.zeros(len(x), len(x[0]), max(lens), len(x[0][0][0])).float()
        for p_i, f_x in enumerate(x):
            end = lens[p_i]
            for f_i, xx in enumerate(f_x):
                x_pad[p_i,f_i,:end] = torch.FloatTensor(xx)
        return x_pad, torch.FloatTensor(y), torch.LongTensor(lens)