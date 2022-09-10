import os
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero
from torch.nn import functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import pickle as pkl
from shutil import copyfile
import numpy as np
from glob import glob
import random
import torch
import scipy.sparse as sp
from torch_geometric.data import Data, HeteroData
from utils.common_utils import setup_seed
from utils.graph_utils import build_graph
from utils.data_utils import load_record
from sklearn.model_selection import StratifiedKFold
from models import GNN, MC_GRU
from trainer import Motality_Trainer
from datasets import PatientDataset
from utils.model_utils import load_model

def train(args):
    # TODO: 把训练代码写到这里
    pass

def load_data(x_path, y_path, static_path):
    x = pkl.load(open(x_path, 'rb'))
    y = pkl.load(open(y_path, 'rb'))
    y = [yy[-1] for yy in y]
    static = pkl.load(open(static_path, 'rb'))
    
    return x, y, static

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)
    # ----------------- Instantiate Dataset ------------------------
    print('#'*20,'Start Loading Data','#'*20)
    x, y, static = load_data(x_path=args.x_path, y_path=args.y_path, static_path=args.static_path)
    print('#'*20,'End Loading Data','#'*20)
    
    kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    kfold_metrics = {}
    for i, (train_set, val_set) in enumerate(kfold.split(x, y)):
        train_dataset = PatientDataset(
            x_data=np.array(x)[train_set],
            y_data=np.array(y)[train_set],
            static_data=np.array(static)[train_set],
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        
        valid_dataset = PatientDataset(
            x_data=np.array(x)[val_set],
            y_data=np.array(y)[val_set],
            static_data=np.array(static)[val_set],
        )
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=valid_dataset.collate_fn)
        
        # ----------------- Instantiate Model --------------------------
        model = load_model(args.model_name, args)
        trainer = Motality_Trainer(
            train_loader=train_loader,
            valid_loader=valid_loader,
            batch_size=args.batch_size,
            num_epochs=args.epoch,
            fold=i,
            log_dir=os.path.join(args.log_dir, 'kfold-'+str(i)),
            lr=args.lr,
            device=device,
            early_stop=args.early_stop,
            model=model,
            monitor=args.monitor,
            save_path=os.path.join(args.ckpt_save_path, 'kfold-'+str(i))
        )
        print('#'*20,"kfold-{}: starting training...".format(i),'#'*20)
        for epoch in range(1, args.epoch + 1):
            continue_train = trainer.train_epoch(epoch)
            if not continue_train:
                break
        
        copyfile(trainer.best_metric_model_path, trainer.best_metric_model_path.replace('.pth', '_'+str(trainer.best_metric)+'.pth'))
        
        for key, value in trainer.metric_all.items():
            if key not in kfold_metrics:
                kfold_metrics[key] = [value]
            else:
                kfold_metrics[key].append(value)
        
        print('#'*20,"kfold-{}: end training".format(i),'#'*20)
    
    for key, value in kfold_metrics.items():
        print('%s: %.4f(%.4f)'%(key, np.mean(value), np.std(value)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--kfold', default=5, type=int)
    parser.add_argument('--early_stop', default=10, type=int)
    parser.add_argument('--kernel_size', default=4, type=int)
    parser.add_argument('--model_name', default='Multi_ConvTransformer', type=str)
    parser.add_argument('--monitor', default='auroc', type=str)
    parser.add_argument('--log_dir', default='/home/jxf/code/Shapelet_GNN/logs/motality/debug', type=str)
    parser.add_argument('--x_path', default='/home/jxf/code/Shapelet_GNN/input/shapelet_filter_0908/x', type=str)
    parser.add_argument('--y_path', default='/home/jxf/code/Shapelet_GNN/input/shapelet_filter_0908/y', type=str)
    parser.add_argument('--static_path', default='/home/jxf/code/Shapelet_GNN/input/shapelet_filter_0908/static', type=str)
    parser.add_argument('--ckpt_save_path', default='/home/jxf/code/Shapelet_GNN/checkpoints/motality/debug', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    args = parser.parse_args()
    main(args)