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
import math
import torch
import time
from utils.metric_utils import print_metrics_binary
from utils.data_utils import get_all_segment,get_selected_segment_id,get_selected_segment,vis_kmedoids
from pathos.multiprocessing import ProcessingPool as Pool
from torch.nn.parallel import DistributedDataParallel as DDP

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
        test_loader=None,
        lr=1e-3,
        early_stop=1e9,
        loss='bce',
        fold=0,
        pretrain_ckpt_path=None,
        ddp=False,
        local_rank=0,
        dist=None,
        decov_w=1000
    ):
        self.device = device
        self.ddp = ddp
        self.dist = dist
        self.early_stop = early_stop
        self.no_lift_count = 0
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.monitor = monitor
        self.save_path = save_path
        self.loss_fn = self.configure_loss(loss)
        self.decov_w = decov_w
        
        if self.ddp:
            self.model = model.to(local_rank)
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            self.loss_fn.to(local_rank)
        else:
            self.model = model.to(self.device)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.pretrain_ckpt_path = pretrain_ckpt_path
        self.log_dir = log_dir
        self.metric_all=None
        self.lr = lr
        self.fold=fold
        self.best_motality_model_path = None
        self.optimizer = self.configure_optimizer()
        
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.tensorwriter = SummaryWriter(log_dir)
        
        os.makedirs(self.save_path, exist_ok=True)
        
        self.best_loss = 1e9
        self.best_metric = -1e9
    
    def train_epoch(self, epoch):
        self.model.train()
        if self.ddp:
            self.train_loader.sampler.set_epoch(epoch)
        train_iterator = tqdm(
            self.train_loader, desc="Fold {}: Epoch {}/{}".format(self.fold, epoch, self.num_epochs), leave=False
        )
        loss_epoch = 0
        decov_epoch = 0
        for x, y, static, lens, _ in train_iterator:
            x = x.to(self.device)
            y = y.to(self.device)
            static = static.to(self.device)
            
            res = self.model(x, lens.cpu(), static)
            pred = res['output']
            
            loss = self.loss_fn(pred, y)
            
            # if 'decov_loss' in res:
            #     base_m = math.floor(math.log(loss.item(), 10))
            #     decov_m = math.floor(math.log(res['decov_loss'].item(), 10))
            #     decov_loss = self.decov_w * res['decov_loss'] * 10 ** (base_m-decov_m)
            #     loss += decov_loss
            #     decov_epoch += decov_loss.item()
            
            self.model.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            if self.ddp:
                self.dist.all_reduce(loss, op=self.dist.ReduceOp.SUM)
                loss /= self.dist.get_world_size()
            loss_epoch += loss.item()  
            
        loss_epoch /= len(self.train_loader)
        decov_epoch /= len(self.train_loader)
        print(f"Fold {self.fold}: Epoch {epoch}:")
        print(f"Train Loss: {loss_epoch:.4f}")
        print(f"Decov Loss: {decov_epoch:.4f}")
        
        self.tensorwriter.add_scalar("train_loss/epoch", loss_epoch, epoch)
        
        if not self.ddp or self.dist.get_rank() == 0:
            eval_loss, eval_metric = self.evaluate(epoch)
            
            if eval_metric[self.monitor] > self.best_metric:
                self.best_metric = eval_metric[self.monitor]
                self.best_metric_model_path = os.path.join(self.save_path, 'best_model.pth')
                if self.ddp:
                    torch.save(self.model.module.state_dict(), self.best_metric_model_path)
                else:
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
                
                pred = self.model(x, lens.cpu(), static)['output']
                
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

    def test(self):
        self.model.eval()
        test_iterator = tqdm(
            self.test_loader, desc="Test", total=len(self.test_loader)
        )
        all_y = []
        all_pred = []
        with torch.no_grad():
            for x, y, static, lens, _ in test_iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                static = static.to(self.device)
                
                pred = self.model(x, lens.cpu(), static)['output']
                
                all_y.append(y)
                all_pred.append(pred)
        
        all_y = torch.cat(all_y, dim=0).squeeze()
        all_pred = torch.cat(all_pred, dim=0).squeeze()
        
        loss = self.loss_fn(all_pred, all_y)
        print(f"Test Loss: {loss:.4f}")
        
        all_y = all_y.cpu().detach().numpy().flatten()
        all_pred = all_pred.cpu().detach().numpy().flatten()
        metrics = print_metrics_binary(all_pred, all_y)
        
        # -----------------------------------------------------------------------------------------------------------------
        print('#'*20,"starting Test Bootstrap...",'#'*20)
        N = len(all_y)
        N_idx = np.arange(N)
        K = 1000

        auroc = []
        auprc = []
        minpse = []
        f1 = []
        
        for _ in tqdm(range(K), desc='Test Bootstrap'):
            # import pdb;pdb.set_trace()
            boot_idx = np.random.choice(N_idx, N, replace=True)
            boot_true = all_y[boot_idx]
            boot_pred = all_pred[boot_idx]
            test_ret = print_metrics_binary(boot_pred, boot_true)
            auroc.append(test_ret['auroc'])
            auprc.append(test_ret['auprc'])
            minpse.append(test_ret['minpse'])
            f1.append(test_ret['f1_score'])
        
        print('auroc: %.4f(%.4f)'%(np.mean(auroc), np.std(auroc)))
        print('auprc: %.4f(%.4f)'%(np.mean(auprc), np.std(auprc)))
        print('minpse: %.4f(%.4f)'%(np.mean(minpse), np.std(minpse)))
        print('f1 score: %.4f(%.4f)'%(np.mean(f1), np.std(f1)))
    
    def gen_frontend_emb(self, save_path, data_loader):
        self.model.eval()
        data_iterator = tqdm(
            data_loader, desc="All Emb", total=len(data_loader)
        )
        
        data_emb = self.get_emb(data_iterator, self.model)
        os.makedirs(save_path, exist_ok=True)
        pkl.dump(data_emb, open(os.path.join(save_path, 'emb.pkl'),'wb'))
        
        
        
    
    def gen_frontend_emb(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.model.eval()
        
        if self.test_loader is not None:
            test_iterator = tqdm(
                self.test_loader, desc="Test", total=len(self.test_loader)
            )
            test_emb = self.get_emb(test_iterator, self.model)
            pkl.dump(test_emb, open(os.path.join(save_path, 'test_emb.pkl'),'wb'))
            
        valid_iterator = tqdm(
            self.valid_loader, desc="Valid", total=len(self.valid_loader)
        )
        valid_emb, valid_y, valid_pred = self.get_emb(valid_iterator, self.model)
        pkl.dump(valid_emb, open(os.path.join(save_path, 'valid_emb.pkl'),'wb'))
        pkl.dump(valid_y, open(os.path.join(save_path, 'valid_y.pkl'),'wb'))
        pkl.dump(valid_pred, open(os.path.join(save_path, 'valid_pred.pkl'),'wb'))
        
        train_iterator = tqdm(
            self.train_loader, desc="Train", total=len(self.train_loader)
        )
        train_emb, train_y, train_pred = self.get_emb(train_iterator, self.model)
        pkl.dump(train_emb, open(os.path.join(save_path, 'train_emb.pkl'),'wb'))
        pkl.dump(train_y, open(os.path.join(save_path, 'train_y.pkl'),'wb'))
        pkl.dump(train_pred, open(os.path.join(save_path, 'train_pred.pkl'),'wb'))
        
        
    def get_emb(self, data_iterator, model):
        all_emb = []
        all_y = []
        all_pred = []
        with torch.no_grad():
            for x, y, static, lens, _ in data_iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                static = static.to(self.device)

                res = model(x, lens, static)
                emb = res['emb']
                pred = res['output']
                
                all_emb.append(emb)
                all_y.append(y)
                all_pred.append(pred)
        
        all_emb = torch.cat(all_emb, dim=0).squeeze()
        all_y = torch.cat(all_y, dim=0).squeeze()
        all_pred = torch.cat(all_pred, dim=0).squeeze()
        
        all_y = all_y.cpu().detach().numpy().flatten()
        all_pred = all_pred.cpu().detach().numpy().flatten()
        all_emb = all_emb.cpu().detach().numpy()
        
        print_metrics_binary(all_pred, all_y)
        
        
        return all_emb, all_y, all_pred
        

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
        
        selected_feature = [13]
        
        pool = Pool(node=len(selected_feature)+1)
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