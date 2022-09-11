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
from tensorboardX import SummaryWriter
from utils.common_utils import setup_seed
from utils.graph_utils import build_graph
from utils.data_utils import load_record, load_ori_data
from sklearn.model_selection import StratifiedKFold
from models import GNN, MC_GRU
from trainer import Shapelet_Trainer
from datasets import RecordDataset

def train(args, data_path):
    # TODO: 把训练代码写到这里
    pass
    


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)
    # ----------------- Instantiate Dataset ------------------------
    print('#'*20,'Start Loading Data','#'*20)
    x, y = load_ori_data(args.ori_data_path)
    
    print('#'*20,'End Loading Data','#'*20)
    
    kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    kfold_metrics = {}
    for i, (train_set, val_set) in enumerate(kfold.split(x, y)):
        graph_path = os.path.join(args.graph_path, 'kfold-'+str(i))
        record_data = load_record(record_path=graph_path)
        
        train_dataset = RecordDataset(
            input_data={key:np.array(value)[train_set] for key, value in record_data.items()}, 
            label_data=np.array(y)[train_set], 
            data_dir=graph_path, 
            type='train', 
            topk=5)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        
        valid_dataset = RecordDataset(
            input_data={key:np.array(value)[val_set] for key, value in record_data.items()}, 
            label_data=np.array(y)[val_set], 
            data_dir=graph_path, 
            type='valid', 
            topk=5)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=valid_dataset.collate_fn)
        
        # ----------------- Instantiate Graph --------------------------
        graph_data_path = os.path.join(graph_path, 'graph.pkl')
        if not os.path.isfile(graph_data_path):
            graph_data = build_graph(graph_path=graph_path)
            pkl.dump(graph_data, open(graph_data_path, 'wb'))
        else:
            print("Found preprocessed graph. Loading that!")
            graph_data = pkl.load(open(graph_data_path, 'rb'))
        
        
        # ----------------- Instantiate Model --------------------------
        graph_model = GNN(hidden_channels=64, out_channels=32)
        graph_model = to_hetero(graph_model, graph_data.metadata(), aggr='sum')
        motality_model = MC_GRU(feature_size=17, input_dim=32, hidden_dim=64, output_dim=1)
        # ----------------- Instantiate Trainer ------------------------
        trainer = Shapelet_Trainer(
            train_loader=train_loader,
            valid_loader=valid_loader,
            batch_size=args.batch_size,
            num_epochs=args.epoch,
            log_dir=os.path.join(args.log_dir, 'kfold-'+str(i)),
            graph_data=graph_data,
            lr=args.lr,
            fold=i,
            monitor=args.monitor,
            device=device,
            graph_model=graph_model,
            motality_model=motality_model,
            save_path=os.path.join(args.ckpt_save_path, 'kfold-'+str(i))
        )
        print('#'*20,"kfold-{}: starting training...".format(i),'#'*20)
        for epoch in range(1, args.epoch + 1):
            trainer.train_epoch(epoch)
        
        copyfile(trainer.best_metric_graph_model_path, trainer.best_metric_graph_model_path.replace('.pth', '_'+ args.monitor +'_'+str(trainer.best_metric)+'.pth'))
        copyfile(trainer.best_metric_motality_model_path, trainer.best_metric_motality_model_path.replace('.pth', '_'+ args.monitor +'_'+str(trainer.best_metric)+'.pth'))
        
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
    parser.add_argument('--log_dir', default='/home/jxf/code/Shapelet_GNN/logs/shapelet/debug', type=str)
    parser.add_argument('--ckpt_save_path', default='/home/jxf/code/Shapelet_GNN/checkpoints/shapelet/debug', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--monitor', default='auroc', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--graph_path', default='/home/jxf/code/Shapelet_GNN/input/graph_data/20220911-01_Multi_ConvTransformer_clstoken_dropout5_lr5_kernel-5_auroc/layer-2/offset-0_segL-12_dist-softdtw_percentile-10', type=str)
    parser.add_argument('--ori_data_path', default='/home/jxf/code/Shapelet_GNN/input/shapelet_filter_0908', type=str)
    args = parser.parse_args()
    main(args)
    
    
    
    
    