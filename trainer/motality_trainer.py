import os
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero
from torch import nn
import copy
from torch.nn import functional as F
from argparse import ArgumentParser
from tqdm import tqdm
from tensorboardX import SummaryWriter
import pickle as pkl
import numpy as np
from glob import glob
import random
import torch
import time
from utils.metric_utils import print_metrics_binary
from utils.data_utils import get_all_segment,get_selected_segment_id,get_selected_segment,vis_kmedoids
from pathos.multiprocessing import ProcessingPool as Pool

class Motality_Trainer(object):
    def __init__(
        self,
        train_loader,
        valid_loader,
        batch_size,
        num_epochs,
        log_dir,
        device,
        model,
        save_path,
        monitor='auprc',
        lr=1e-3,
        early_stop=1e9,
        loss='bce',
        fold=0,
        pretrain_ckpt_path=None,
    ):
        self.device = device
        self.early_stop = early_stop
        self.no_lift_count = 0
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.monitor = monitor
        self.save_path = save_path
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.pretrain_ckpt_path = pretrain_ckpt_path
        self.log_dir = log_dir
        self.metric_all=None
        self.lr = lr
        self.fold=fold
        self.best_motality_model_path = None
        self.loss_fn = self.configure_loss(loss)
        self.optimizer = self.configure_optimizer()
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.tensorwriter = SummaryWriter(log_dir)
        
        os.makedirs(self.save_path, exist_ok=True)
        
        self.best_loss = 1e9
        self.best_metric = -1e9
    
    def train_epoch(self, epoch):
        self.model.train()
        train_iterator = tqdm(
            self.train_loader, desc="Fold {}: Epoch {}/{}".format(self.fold, epoch, self.num_epochs), leave=False
        )
        loss_epoch = 0
        for x, y, static, lens, _ in train_iterator:
            x = x.to(self.device)
            y = y.to(self.device)
            static = static.to(self.device)
            
            pred = self.model(x, lens, static)['output']
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss.item()
            
        loss_epoch /= len(self.train_loader)
        print(f"Fold {self.fold}: Epoch {epoch}:")
        print(f"Train Loss: {loss_epoch:.4f}")
        self.tensorwriter.add_scalar("train_loss/epoch", loss_epoch, epoch)
        eval_loss, eval_metric = self.evaluate(epoch)
        
        if eval_metric[self.monitor] > self.best_metric:
            self.best_metric = eval_metric[self.monitor]
            self.best_metric_model_path = os.path.join(self.save_path, 'best_model.pth')
            torch.save(self.model.state_dict(), self.best_metric_model_path)
            self.metric_all = eval_metric
            self.no_lift_count = 0
        else:
            self.no_lift_count += 1
            if self.no_lift_count > self.early_stop:
                return False
        
        print('-'*100)
        print('Epoch {}, best eval {}: {}'.format(epoch, self.monitor, self.best_metric))
        print('-'*100)
        return True
    
    __call__ = train_epoch
    
    def evaluate(self, epoch):
        self.model.eval()
        eval_iterator = tqdm(
            self.valid_loader, desc="Evaluation", total=len(self.valid_loader)
        )
        all_y = []
        all_pred = []
        with torch.no_grad():
            for x, y, static, lens, _ in eval_iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                static = static.to(self.device)
                
                pred = self.model(x, lens, static)['output']
                
                all_y.append(y)
                all_pred.append(pred)
        
        all_y = torch.cat(all_y, dim=0).squeeze()
        all_pred = torch.cat(all_pred, dim=0).squeeze()
        
        loss = self.loss_fn(all_pred, all_y)
        print(f"Epoch {epoch}:")
        print(f"Eval Loss: {loss:.4f}")
        
        metrics = print_metrics_binary(all_pred.cpu().detach().numpy().flatten(), all_y.cpu().detach().numpy().flatten())
        self.tensorwriter.add_scalar("eval_loss/epoch", loss, epoch)
        self.tensorwriter.add_scalar("eval_auprc/epoch", metrics['auprc'], epoch)
        self.tensorwriter.add_scalar("eval_minpse/epoch", metrics['minpse'], epoch)
        self.tensorwriter.add_scalar("eval_auroc/epoch", metrics['auroc'], epoch)
        self.tensorwriter.add_scalar("eval_prec0/epoch", metrics['prec0'], epoch)
        self.tensorwriter.add_scalar("eval_acc/epoch", metrics['acc'], epoch)
        self.tensorwriter.add_scalar("eval_prec1/epoch", metrics['prec1'], epoch)
        self.tensorwriter.add_scalar("eval_rec0/epoch", metrics['rec0'], epoch)
        self.tensorwriter.add_scalar("eval_rec1/epoch", metrics['rec1'], epoch)
        self.tensorwriter.add_scalar("eval_f1_score/epoch", metrics['f1_score'], epoch)
        
        return loss, metrics

    def gen_shapelet(self, model, fold, cluster_num, threshold_rate, medical_idx, save_path):
        self.model.eval()
        train_iterator = tqdm(
            self.train_loader, desc="Fold {}: Generate Shapelet Eval".format(fold), total=len(self.train_loader)
        )
        all_y = []
        all_pred = []
        all_attn = []
        all_pdid = []
        all_x = []
        all_lens = []
        with torch.no_grad():
            for x, y, static, lens, pdid in train_iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                static = static.to(self.device)
                
                res = self.model(x, lens, static)
                pred = res['output']
                attn = res['attns']
                
                all_y.append(y)
                all_pred.append(pred)
                all_pdid.append(pdid)
                all_x += x.cpu().detach().tolist()
                all_attn+=attn.cpu().detach().tolist()
                all_lens.append(lens)
        
        all_y = torch.cat(all_y, dim=0).squeeze().cpu().detach().numpy()
        all_pred = torch.cat(all_pred, dim=0).squeeze().cpu().detach().numpy()
        all_pdid = torch.cat(all_pdid, dim=0).squeeze().cpu().detach().numpy()
        all_lens = torch.cat(all_lens, dim=0).squeeze().cpu().detach().numpy()
        
        pkl.dump(all_x, open(os.path.join(save_path, 'x_train'), 'wb'))
        pkl.dump(all_pdid, open(os.path.join(save_path, 'pdid_train'), 'wb'))
        
        metrics = print_metrics_binary(all_pred.flatten(), all_y.flatten())
        kernel_size = model.kernel_size
        conv_layer_num = len(model.num_channels)
        
        
        selected_feature = list(range(len(all_x[0][0])))
        
        selected_x = []
        selected_attn = []
        for fi in selected_feature:
            selected_x.append([copy.deepcopy(np.array(xx)[:, fi]) for xx in all_x])
            selected_attn.append([copy.deepcopy(np.array(aa)[fi, :, :]) for aa in all_attn])
        
        pool = Pool(node=len(selected_feature))
        results = pool.map(self.gen_shapelet_for_feature, 
                           selected_feature, 
                           selected_x, 
                           selected_attn, 
                           [all_pdid] * len(selected_feature), 
                           [all_lens] * len(selected_feature), 
                           [kernel_size] * len(selected_feature), 
                           [conv_layer_num] * len(selected_feature), 
                           [cluster_num] * len(selected_feature), 
                           [threshold_rate] * len(selected_feature), 
                           [copy.deepcopy(medical_idx)] * len(selected_feature), 
                           [save_path] * len(selected_feature)
                           )
        
        
        # all_x = [np.array(xx)[:, selected_feature] for xx in all_x]
        # all_attn = [np.array(aa)[selected_feature, :, :] for aa in all_attn]
        
        
        
        
            
    def configure_loss(self, loss_name):
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'bce':
            return nn.BCELoss()
        else:
            raise ValueError("Invalid Loss Type!")
        
    def configure_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    
    @staticmethod
    def gen_shapelet_for_feature(selected_feature, all_x, all_attn, all_pdid, all_lens, kernel_size, conv_layer_num, cluster_num, threshold_rate, medical_idx, save_path):
        # get all segments
        all_segments = get_all_segment(x=all_x, conv_layer_num=conv_layer_num, kernel_size=kernel_size)
        
        # get selected segment id
        all_select_seg_idx = get_selected_segment_id(all_segments=all_segments, attns=all_attn, threshold_rate=threshold_rate)
        
        # get selected segment
        selected_segments_all, selected_segments_all_idx = get_selected_segment(all_select_seg_idx=all_select_seg_idx, all_segments=all_segments)
        
        print('******************* start clustering *******************')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for i, selected_seg_all_layer in enumerate(selected_segments_all):
            start = time.perf_counter()
            print('feature {} :{} segment num: {}'.format(selected_feature, i, len(selected_seg_all_layer)))
            
            x_vis = [xx[:ll] for (xx, ll) in zip(all_x, all_lens)]
            vis_kmedoids(np.array(selected_seg_all_layer), 
                         np.array(selected_segments_all_idx[i]), 
                         x_vis, 
                         all_pdid, 
                         cluster_num,
                         os.path.join(save_path, '{}_{}_train_layer{}.pdf'.format(selected_feature, medical_idx[selected_feature], i)),
                         os.path.join(save_path, '{}_{}_train_layer{}.pkl'.format(selected_feature, medical_idx[selected_feature], i))
                         )
            end = time.perf_counter()
        print('Running time for feature %d layer %d: %s Seconds'%(selected_feature, i, end-start))