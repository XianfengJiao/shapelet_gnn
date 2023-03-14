import os
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
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
from utils.data_utils import load_record
from sklearn.model_selection import StratifiedKFold
from models import GNN, MC_GRU
from trainer import Motality_Trainer
from datasets import PatientDataset
from utils.model_utils import load_model
import torch.distributed as dist

def train(trainer, train_loader, valid_loader, args):
    # TODO: 把训练代码写到这里
    pass

def load_data(data_path, pdid_path=None):
    
    
    train_x = pkl.load(open(os.path.join(data_path, 'train_x.pkl'), 'rb'))
    train_y = pkl.load(open(os.path.join(data_path, 'train_y.pkl'), 'rb'))
    # train_y = [yy[-1] for yy in train_y]
    train_static = pkl.load(open(os.path.join(data_path, 'train_static.pkl'), 'rb'))
    train_pdid = np.array(range(len(train_x)))
    
    val_x = pkl.load(open(os.path.join(data_path, 'val_x.pkl'), 'rb'))
    val_y = pkl.load(open(os.path.join(data_path, 'val_y.pkl'), 'rb'))
    # val_y = [yy[-1] for yy in val_y]
    val_static = pkl.load(open(os.path.join(data_path, 'val_static.pkl'), 'rb'))
    val_pdid = np.array(range(len(val_x)))
    val_pdid = np.array(range(len(train_x), len(train_x)+len(val_x)))
    
    test_x = pkl.load(open(os.path.join(data_path, 'test_x.pkl'), 'rb'))
    test_y = pkl.load(open(os.path.join(data_path, 'test_y.pkl'), 'rb'))
    # test_y = [yy[-1] for yy in test_y]
    test_static = pkl.load(open(os.path.join(data_path, 'test_static.pkl'), 'rb'))
    test_pdid = np.array(range(len(train_x)+len(val_x), len(train_x)+len(val_x)+len(test_x)))
        
    return train_x, train_y, train_static, train_pdid, val_x, val_y, val_static, val_pdid, test_x, test_y, test_static, test_pdid

def main(args):
    # ----------------- init ddp ------------------------
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    setup_seed(args.seed)
    # ----------------- Instantiate Dataset ------------------------
    print('#'*20,'Start Loading Data','#'*20)
    train_x, train_y, train_static, train_pdid, val_x, val_y, val_static, val_pdid, test_x, test_y, test_static, test_pdid = load_data(data_path = args.data_path)
    if args.lens_path != '':
        lens = pkl.load(open(args.lens_path, 'rb'))
    print('#'*20,'End Loading Data','#'*20)

    train_dataset = PatientDataset(
        x_data=np.array(train_x, dtype=object),
        y_data=np.array(train_y),
        static_data=np.array(train_static),
        pdid_data=np.array(train_pdid),
        lens_data=None,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn, sampler=train_sampler)
    
    valid_dataset = PatientDataset(
        x_data=np.array(val_x, dtype=object),
        y_data=np.array(val_y),
        static_data=np.array(val_static),
        pdid_data=np.array(val_pdid),
        lens_data=None,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=not args.only_emb, collate_fn=valid_dataset.collate_fn)
    
    test_dataset = PatientDataset(
        x_data=np.array(test_x, dtype=object),
        y_data=np.array(test_y),
        static_data=np.array(test_static),
        pdid_data=np.array(test_pdid),
        lens_data=None,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=not args.only_emb, collate_fn=test_dataset.collate_fn)
    
    # ----------------- Instantiate Model --------------------------
    model = load_model(args.model_name, args)
    
    # ----------------- Load Checkpoint --------------------------
    if args.pretrain_model_path != '':
        print(f'Load checkpoint from {args.pretrain_model_path}')
        model.load_state_dict(torch.load(args.pretrain_model_path, map_location=torch.device(device)))
        
    # ----------------- Instantiate Trainer ------------------------
    trainer = Motality_Trainer(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        batch_size=args.batch_size,
        num_epochs=args.epoch,
        fold=0,
        log_dir=os.path.join(args.log_dir, 'kfold-0'),
        lr=args.lr,
        device=device,
        early_stop=args.early_stop,
        model=model,
        monitor=args.monitor,
        save_path=os.path.join(args.ckpt_save_path, 'kfold-0'),
        decov_w=args.decov_w,
        ddp=True,
        local_rank=local_rank,
        dist=dist,
    )
    # ----------------- Generate Shapelet ------------------------
    # if args.gen_shapelet:
    #     pretrain_model_path_fold = os.path.join(args.pretrain_model_path, 'kfold-'+str(i), 'best_model.pth')
    #     model.load_state_dict(torch.load(pretrain_model_path_fold))
    #     trainer.gen_shapelet(model=model,
    #                          fold=i,
    #                          cluster_num=args.cluster_num,
    #                          threshold_rate=args.threshold_rate,
    #                          medical_idx=['Cl','CO2CP','WBC','Hb','Urea','Ca','K','Na','Scr','P','Albumin','hs-CRP','Glucose','Appetite','Weight','SBP','DBP'], 
    #                         #  medical_idx=['Scr'], 
    #                          save_path=trainer.log_dir)
    #     continue
    
    # ----------------- Only Testing ------------------------
    if args.only_test:
        print('#'*20,"start testing...",'#'*20)
        trainer.test()
        print('#'*20,"end testing",'#'*20)
        exit(0)
    
    if args.only_emb:
        print('#'*20,"start generating embedding...",'#'*20)
        trainer.gen_frontend_emb(save_path=os.path.join(args.log_dir, 'saved_emb'))
        print('#'*20,"end generating embedding...",'#'*20)
        
    
    # ----------------- Start Training ------------------------
    print('#'*20,"starting training...",'#'*20)
    for epoch in range(1, args.epoch + 1):
        continue_train = trainer.train_epoch(epoch)
        if not continue_train:
            break
    print('#'*20,"end training",'#'*20)
    # print('#'*40)
    # print('best results: ')
    # for key, value in trainer.metric_all.items():
    #     print(key, ': ', value)
    
    # dist.barrier()
    if dist.get_rank() == 0:
        copyfile(trainer.best_metric_model_path, trainer.best_metric_model_path.replace('.pth', '_'+args.monitor+'_'+str(trainer.best_metric)+'.pth'))
        print('#'*20,"starting testing...",'#'*20)
        trainer.test()
        print('#'*20,"end testing",'#'*20)
    
    # copyfile(trainer.best_metric_model_path, trainer.best_metric_model_path.replace('.pth', '_'+args.monitor+'_'+str(trainer.best_metric)+'.pth'))
    # print('#'*20,"starting testing...",'#'*20)
    # trainer.test()
    # print('#'*20,"end testing",'#'*20)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--kfold', default=5, type=int)
    parser.add_argument('--cluster_num', default=10, type=int)
    parser.add_argument('--threshold_rate', default=98, type=int)
    parser.add_argument('--early_stop', default=15, type=int)
    parser.add_argument('--kernel_size', default=5, type=int)
    parser.add_argument('--input_dim', default=76, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--model_name', default='ConCare_mimiic', type=str)
    parser.add_argument('--monitor', default='auroc', type=str)
    parser.add_argument("--pretrain_model_path", default='', type=str)
    parser.add_argument("--gen_shapelet", action='store_true', help="only to generate shapelet.",)
    parser.add_argument("--only_test", action='store_true', help="only to test.",)
    parser.add_argument("--only_emb", action='store_true', help="only to generate embedding.",)
    parser.add_argument("--fix_fold",  default=-1, type=int)
    parser.add_argument('--log_dir', default='/home/jxf/code/Shapelet_GNN/logs/motality/mimiic/debug', type=str)
    parser.add_argument('--data_path', default='/home/jxf/code/Shapelet_GNN/input/mimiic/', type=str)
    parser.add_argument('--train_x_path', default='/home/jxf/code/Shapelet_GNN/input/mimiic/train_x.pkl', type=str)
    parser.add_argument('--train_y_path', default='/home/jxf/code/Shapelet_GNN/input/mimiic/train_y.pkl', type=str)
    parser.add_argument('--train_static_path', default='/home/jxf/code/Shapelet_GNN/input/mimiic/train_static.pkl', type=str)
    parser.add_argument('--val_x_path', default='/home/jxf/code/Shapelet_GNN/input/mimiic/val_x.pkl', type=str)
    parser.add_argument('--val_y_path', default='/home/jxf/code/Shapelet_GNN/input/mimiic/val_y.pkl', type=str)
    parser.add_argument('--val_static_path', default='/home/jxf/code/Shapelet_GNN/input/mimiic/val_static.pkl', type=str)
    parser.add_argument('--test_x_path', default='/home/jxf/code/Shapelet_GNN/input/mimiic/test_x.pkl', type=str)
    parser.add_argument('--test_y_path', default='/home/jxf/code/Shapelet_GNN/input/mimiic/test_y.pkl', type=str)
    parser.add_argument('--test_static_path', default='/home/jxf/code/Shapelet_GNN/input/mimiic/test_static.pkl', type=str)
    parser.add_argument('--lens_path', default='', type=str)
    parser.add_argument('--pdid_path', default='', type=str)
    parser.add_argument('--ckpt_save_path', default='/home/jxf/code/Shapelet_GNN/checkpoints/motality/mimiic/debug', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--demo_dim', default=12, type=int)
    parser.add_argument('--num_channels', default='4,8,16,32', type=str)
    parser.add_argument('--layer_num', default=2, type=int)
    parser.add_argument('--keep_prob', default=0.5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--decov_w', default=1000, type=float)
    args = parser.parse_args()
    
    args.num_channels = [int(n) for n in args.num_channels.split(',')]
    main(args)