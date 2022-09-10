import os
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero
from torch import nn
from torch.nn import functional as F
from argparse import ArgumentParser
from tqdm import tqdm
from tensorboardX import SummaryWriter
import pickle as pkl
import numpy as np
from glob import glob
import random
import torch
from utils.metric_utils import print_metrics_binary

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
        for x, y, static, lens in train_iterator:
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
            self.best_metric_model_path = os.path.join(self.save_path, 'best_' + self.monitor +'_model.pth')
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
            for x, y, static, lens in eval_iterator:
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
    
    def configure_loss(self, loss_name):
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'bce':
            return nn.BCELoss()
        else:
            raise ValueError("Invalid Loss Type!")
        
    def configure_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)