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

class Shapelet_Trainer(object):
    def __init__(
        self,
        train_loader,
        valid_loader,
        batch_size,
        num_epochs,
        log_dir,
        device,
        motality_model,
        graph_model,
        save_path,
        graph_data,
        fold=0,
        monitor='auprc',
        lr=1e-3,
        loss='bce',
        pretrain_ckpt_path=None,
    ):
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.monitor = monitor
        self.fold=fold
        self.save_path = save_path
        self.motality_model = motality_model.to(self.device)
        self.graph_model = graph_model.to(self.device)
        self.graph_data = graph_data.to(self.device)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.pretrain_ckpt_path = pretrain_ckpt_path
        self.log_dir = log_dir
        self.metric_all=None
        self.lr = lr
        self.best_motality_model_path = None
        self.loss_fn = self.configure_loss(loss)
        self.optimizer = self.configure_optimizer()
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.tensorwriter = SummaryWriter(log_dir)
        
        os.makedirs(self.save_path, exist_ok=True)
        
        self.best_loss = 1e9
        self.best_metric = -1e9
        
        # Initialize lazy modules.
        with torch.no_grad():  
            self.graph_model(self.graph_data.x_dict, self.graph_data.edge_index_dict)
    
    def train_epoch(self, epoch):
        self.motality_model.train()
        self.graph_model.train()
        train_iterator = tqdm(
            self.train_loader, desc="Fold {}: Epoch {}/{}".format(self.fold, epoch, self.num_epochs), leave=False
        )
        loss_epoch = 0
        for x, y, lens in train_iterator:
            x = x.to(self.device)
            y = y.to(self.device)
            
            node_embeddings = self.graph_model(self.graph_data.x_dict, self.graph_data.edge_index_dict)
            
            # 将 x 选择的 node 表示进行相加
            x_converted = torch.Tensor().to(self.device)
            feature_size = x.shape[1]
            
            for f_i in range(feature_size):
                f_converted = torch.einsum("sh,brs->brh", node_embeddings['f'+str(f_i)], x[:,f_i,:,:])
                x_converted = torch.cat((x_converted, f_converted.unsqueeze(1)), dim=1)
            
            pred = self.motality_model(x_converted, lens)
            loss = self.loss_fn(pred, y)
            self.motality_model.zero_grad()
            self.graph_model.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_epoch += loss.item()
            
        loss_epoch /= len(self.train_loader)
        # Print epoch stats
        print(f"Fold {self.fold}: Epoch {epoch}:")
        print(f"Train Loss: {loss_epoch:.4f}")
        self.tensorwriter.add_scalar("train_loss/epoch", loss_epoch, epoch)
        
        
        eval_loss, eval_metric = self.evaluate(epoch)
        
        if eval_metric[self.monitor] > self.best_metric:
            self.best_metric = eval_metric[self.monitor]
            self.best_metric_graph_model_path = os.path.join(self.save_path, 'best_graph_model.pth')
            self.best_metric_motality_model_path = os.path.join(self.save_path, 'best_motality_model.pth')
            torch.save(self.graph_model.state_dict(), self.best_metric_graph_model_path)
            torch.save(self.motality_model.state_dict(), self.best_metric_motality_model_path)
            self.metric_all = eval_metric
        
        print('-'*100)
        print('Epoch {}, best eval {}: {}'.format(epoch, self.monitor, self.best_metric))
        print('-'*100)
            
    
    __call__ = train_epoch
    
    def evaluate(self, epoch):
        self.motality_model.eval()
        self.graph_model.eval()
        
        eval_iterator = tqdm(
            self.valid_loader, desc="Evaluation", total=len(self.valid_loader)
        )
        all_y = []
        all_pred = []
        with torch.no_grad():
            for x, y, lens in eval_iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                
                node_embeddings = self.graph_model(self.graph_data.x_dict, self.graph_data.edge_index_dict)
                
                # 将 x 选择的 node 表示进行相加
                x_converted = torch.Tensor().to(self.device)
                feature_size = x.shape[1]
                for f_i in range(feature_size):
                    f_converted = torch.einsum("sh,brs->brh", node_embeddings['f'+str(f_i)], x[:,f_i,:,:])
                    x_converted = torch.cat((x_converted, f_converted.unsqueeze(1)), dim=1)
                
                pred = self.motality_model(x_converted, lens)

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
    
        

    def inference(self, motality_model, test_loader):
        pass

    def configure_loss(self, loss_name):
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'bce':
            return nn.BCELoss()
        else:
            raise ValueError("Invalid Loss Type!")

    def configure_optimizer(self):
        return torch.optim.Adam([
                {'params': self.graph_model.parameters()},
                {'params': self.motality_model.parameters()}
                ], 
                lr=1e-3)