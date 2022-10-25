import os
from re import X
import sys
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import gc
from torch.utils.data import Dataset

class MergeDataset(Dataset):
    def __init__(self, labtest_data, static_data, pdid, labtest_lens_data, shape_data, label_data, data_dir, type, topk=5):
        preprocessed_path = os.path.join(data_dir, type+'_merge_'+'preprocessed.pkl')
        os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
        
        if type(labtest_lens_data) == type(None):
            labtest_lens_data = [len(i) for i in labtest_data]

        labtest_data, static_data, pdid, labtest_lens_data, shape_data, label_data = self.filter_data(labtest_data, static_data, pdid, labtest_lens_data, shape_data, label_data)
        
        
        if not os.path.isfile(preprocessed_path):
            patient_size = len(shape_data['f0'])
            feature_size = len(shape_data)
            shape_preprocessed = [[[] for _ in range(feature_size)] for _ in range(patient_size)]
            # feature_size x patient_size x record_size x shapelet_size
            for key, x in shape_data.items():
                i = int(key.replace('f',''))
                for p_i, p_x in enumerate(x):
                    shape_preprocessed[p_i][i] = self.process_data_per_patient(p_x, topk=topk)
            
            pkl.dump(shape_preprocessed, open(preprocessed_path, 'wb'))
        else:
            # -------------------------- Load Preprocessed DATA --------------------------
            print("Found preprocessed record data. Loading that!")
            shape_preprocessed = pkl.load(open(preprocessed_path, 'rb'))
            
        # return patient_size x feature_size x record_size x shapelet_size
        self.data = list(zip(labtest_data, static_data, pdid, labtest_lens_data, shape_preprocessed, label_data))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def filter_data(self, labtest, static, pdid, lens, shape_data, label):
        label = [ll for (ll, dd) in zip(label, shape_data['f0']) if len(dd) > 0]
        labtest = [ll for (ll, dd) in zip(labtest, shape_data['f0']) if len(dd) > 0]
        static = [ll for (ll, dd) in zip(static, shape_data['f0']) if len(dd) > 0]
        pdid = [ll for (ll, dd) in zip(pdid, shape_data['f0']) if len(dd) > 0]
        lens = [ll for (ll, dd) in zip(lens, shape_data['f0']) if len(dd) > 0]
        
        for key in shape_data.keys():
            shape_data[key] = [dd for dd in shape_data[key] if len(dd) > 0]
        
        return labtest, static, pdid, lens, shape_data, label
    
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
        labtest_data, static_data, pdid, labtest_lens, shape_data, label_data = zip(*dataset)
        
        shape_lens = [len(d[0]) for d in shape_data]
        shape_pad = torch.zeros(len(shape_data), len(shape_data[0]), max(shape_lens), len(shape_data[0][0][0])).float()
        for p_i, f_x in enumerate(shape_data):
            end = shape_lens[p_i]
            for f_i, xx in enumerate(f_x):
                shape_pad[p_i,f_i,:end] = torch.FloatTensor(xx)
        
        labtest_pad = torch.zeros(len(labtest_data), max(labtest_lens), len(labtest_data[0][0])).float()
        for i, xx in enumerate(labtest_pad):
            end = labtest_lens[i]
            labtest_pad[i,:end] = torch.FloatTensor(np.array(xx, dtype=float)).unsqueeze(1) if len(np.array(xx, dtype=float).shape) == 1 else torch.FloatTensor(np.array(xx, dtype=float))
            
        return labtest_pad, torch.FloatTensor(static_data), torch.LongTensor(pdid), torch.LongTensor(labtest_lens), shape_pad, torch.LongTensor(shape_lens), torch.FloatTensor(label_data)