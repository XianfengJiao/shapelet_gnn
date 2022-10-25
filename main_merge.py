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
from utils.model_utils import load_model
from sklearn.model_selection import StratifiedKFold
from models import GNN, MC_GRU, MC_GRU_motality, MLP_merge
from trainer import Merge_Trainer
from datasets import MergeDataset

def load_data(x_path, y_path, static_path, pdid_path):
    x = pkl.load(open(x_path, 'rb'))
    y = pkl.load(open(y_path, 'rb'))
    y = [yy[-1] for yy in y]
    static = pkl.load(open(static_path, 'rb'))
    if 'challenge' in static_path or 'hm' in static_path:
        if type(static) == torch.Tensor:
            static = np.array([np.array(ss[-1]) for ss in static])
        else:
            static = [ss[-1] for ss in static]
    if pdid_path != '':
        pdid = pkl.load(open(pdid_path, 'rb'))
    else:
        pdid = np.array(range(len(x)))
    return x, y, static, pdid



def train(args, data_path):
    # TODO: 把训练代码写到这里
    pass


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_seed(args.seed)
    # ----------------- Instantiate Dataset ------------------------
    print('#'*20,'Start Loading Data','#'*20)
    # TODO: load motality data
    x, y, static, pdid = load_data(x_path=args.x_path, y_path=args.y_path, static_path=args.static_path, pdid_path=args.pdid_path)
    if args.lens_path != '':
        lens = pkl.load(open(args.lens_path, 'rb'))
    
    print('#'*20,'End Loading Data','#'*20)
    
    kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    kfold_metrics = {}
    for i, (train_set, val_set) in enumerate(kfold.split(x, y)):
        graph_path = os.path.join(args.graph_path, 'kfold-'+str(i))
        record_data = load_record(record_path=graph_path)
        
        train_dataset = MergeDataset(
            labtest_data=np.array(x, dtype=object)[train_set],
            static_data=np.array(static)[train_set],
            pdid=np.array(pdid)[train_set],
            labtest_lens_data=np.array(lens)[train_set] if args.lens_path != '' else None,
            shape_data={key:np.array(value)[train_set] for key, value in record_data.items()}, 
            label_data=np.array(y)[train_set], 
            data_dir=graph_path, 
            type='train', 
            topk=5)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        
        valid_dataset = MergeDataset(
            labtest_data=np.array(x, dtype=object)[train_set],
            static_data=np.array(static)[train_set],
            pdid=np.array(pdid)[train_set],
            labtest_lens_data=np.array(lens)[train_set] if args.lens_path != '' else None,
            shape_data={key:np.array(value)[val_set] for key, value in record_data.items()}, 
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
        graph_model = load_model(args.graph_model_name, args)
        graph_model = to_hetero(graph_model, graph_data.metadata(), aggr='sum')
        shape_motality_model = load_model(args.shapelet_motality_model_name, args)
        labtest_motality_model = load_model(args.labtest_motality_model_name, args)
        merge_model = MLP_merge()
        
        # 获取到该 fold 上的预训练模型
        # get labtest motality model ckpt
        labtest_motality_model.load_state_dict(torch.load(
            os.path.join(args.labtest_motality_ckpt_path, 'kfold-'+str(i), 'best_model.pth')
            ))
        # get shapelet motality model ckpt
        shape_motality_model.load_state_dict(torch.load(
            os.path.join(args.shape_motality_ckpt_path, 'kfold-'+str(i), 'best_model.pth')
            ))
        # get gnn model ckpt
        graph_model.load_state_dict(torch.load(
            os.path.join(args.graph_ckpt_path, 'kfold-'+str(i), 'best_model.pth')
            ))
        
        # ----------------- Instantiate Trainer ------------------------
        trainer = Merge_Trainer(
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
            shape_motality_model=shape_motality_model,
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
    parser.add_argument('--log_dir', default='/home/jxf/code/Shapelet_GNN/logs/merge/debug', type=str)
    parser.add_argument('--graph_model_name', default='', type=str)
    parser.add_argument('--labtest_motality_model_name', default='', type=str)
    parser.add_argument('--shapelet_motality_model_name', default='', type=str)
    parser.add_argument('--graph_ckpt_path', default='', type=str)
    parser.add_argument('--labtest_motality_ckpt_path', default='', type=str)
    parser.add_argument('--shapelet_motality_ckpt_path', default='', type=str)
    parser.add_argument('--ckpt_save_path', default='/home/jxf/code/Shapelet_GNN/checkpoints/merge/debug', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--monitor', default='auroc', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--graph_path', default='/home/jxf/code/Shapelet_GNN/input/graph_data/20220911-01_Multi_ConvTransformer_clstoken_dropout5_lr5_kernel-5_auroc/layer-2/offset-0_segL-12_dist-softdtw_percentile-10', type=str)
    parser.add_argument('--x_path', default='/home/jxf/code/Shapelet_GNN/input/ckd_shapelet_filter_0908/x', type=str)
    parser.add_argument('--y_path', default='/home/jxf/code/Shapelet_GNN/input/ckd_shapelet_filter_0908/y', type=str)
    parser.add_argument('--lens_path', default='', type=str)
    parser.add_argument('--static_path', default='/home/jxf/code/Shapelet_GNN/input/ckd_shapelet_filter_0908/static', type=str)
    parser.add_argument('--pdid_path', default='', type=str)
    args = parser.parse_args()
    main(args)
    
    
    
    
    