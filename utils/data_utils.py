import os
import pickle as pkl
from glob import glob
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
import torch.utils.data as data
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn_extra.cluster import KMedoids
import tslearn.metrics as metrics
from tslearn.clustering import silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.generators import random_walks
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from math import ceil
import time


def load_ori_data(ori_data_path):
    x = pkl.load(open(os.path.join(ori_data_path, 'x'), 'rb'))
    y = pkl.load(open(os.path.join(ori_data_path, 'y'), 'rb'))
    y = [yy[-1] for yy in y]
    return x, y

def load_record(record_path):
    record_dict = {}
    for record_fp in glob(os.path.join(record_path, '*_sdist_all.pkl')):
        i = os.path.basename(record_fp).split('-')[1].split('_')[0]
        record_dict['f'+i] = pkl.load(open(record_fp, 'rb'))
    
    return record_dict

def vis_kmedoids(seg, seg_index, x_all, pdid_all, num_cluster, vis_save_path,  pkl_save_path):
    plt.rcParams['figure.figsize'] = '70, 340'
    # 声明precomputed自定义相似度计算方法
    km = KMedoids(n_clusters= num_cluster, random_state=0,metric="precomputed")
    # 采用tslearn中的DTW系列及变种算法计算相似度，生成距离矩阵dists
    # dists = metrics.cdist_dtw(X) # dba + dtw
    dists = metrics.cdist_soft_dtw_normalized(seg,gamma=0.5) # softdtw
    y_pred = km.fit_predict(dists)
    np.fill_diagonal(dists,0)
    score = silhouette_score(dists,y_pred,metric="precomputed")
    shapelets = []
    print(seg.shape)
    print(y_pred.shape)
    print("silhouette_score: " + str(score))
    for yi in range(num_cluster):
        plt.subplot(num_cluster, 2, yi * 2 + 1)
        for xx in seg[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.3)
        # 注意这里的_cluster_centers要写成X[km.medoid_indices_[yi]]，因为你是precomputed，源码里面当precomputed时_cluster_centers等于None
        plt.plot(seg[km.medoid_indices_[yi]], "r-")
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                transform=plt.gca().transAxes)
        select_pdid = pdid_all[seg_index[km.medoid_indices_[yi]][0]]
        plt.title("pdid: "+str(select_pdid), fontsize=50)
        # if yi == 1:
        #     plt.title("KMedoids" + " + DBA-DTW")
        
        plt.subplot(num_cluster, 2, yi * 2 + 2)
        select_x = x_all[seg_index[km.medoid_indices_[yi]][0]]
        select_pdid = pdid_all[seg_index[km.medoid_indices_[yi]][0]]
        plt.plot(range(len(select_x)), select_x)
        plt.title("pdid: "+str(select_pdid), fontsize=50)
        plt.axvline(seg_index[km.medoid_indices_[yi]][1], c='red')
        plt.axvline(seg_index[km.medoid_indices_[yi]][1] + len(seg[km.medoid_indices_[yi]]) - 1, c='red')
        shapelets.append({
            'pdid': select_pdid,
            'sta': seg_index[km.medoid_indices_[yi]][1],
            'end': seg_index[km.medoid_indices_[yi]][1] + len(seg[km.medoid_indices_[yi]]) - 1
        })
        
    plt.tight_layout()
    plt.savefig(vis_save_path)
    plt.show()
    plt.clf()
    pkl.dump(shapelets, open(pkl_save_path, 'wb'))

def get_all_segment(x, conv_layer_num, kernel_size):
    all_segments = []

    for select_x in x:
        tmp_segments = [[] for _ in range(conv_layer_num)]
        for i in range(conv_layer_num):
            sta_i = (kernel_size - 1) * (i + 1)
            for s_i in range(sta_i, len(select_x)):
                if select_x[s_i] == 0:
                    break
                s_len = (kernel_size - 1) * (i + 1) + 1
                tmp_segments[i].append(select_x[s_i - s_len + 1: s_i + 1])
        
        all_segments.append(tmp_segments)
    return all_segments

def get_selected_segment_id(all_segments, attns, threshold_rate):
    all_select_seg_idx = []

    for p_i, p_seg in enumerate(all_segments):
        p_attn = attns[p_i]
        tmp_select_seg = []
        for layer_i, layer_attn in enumerate(p_attn):
            try:
                tmp_attn = layer_attn[:np.flatnonzero(layer_attn == 0)[0]]
            except (IndexError):
                tmp_attn = layer_attn
            
            tmp_attn = tmp_attn[-len(all_segments[p_i][layer_i]):]
            
            threshold = np.percentile(tmp_attn, threshold_rate)
            tmp_select_seg.append(np.where(tmp_attn >= threshold)[0])
        all_select_seg_idx.append(tmp_select_seg)
    
    return all_select_seg_idx

def get_selected_segment(all_select_seg_idx, all_segments):
    selected_segments_all = [[] for _ in range(len(all_segments[0]))]
    selected_segments_all_idx = [[] for _ in range(len(all_segments[0]))]
    for p_i, (segment_idx, segments) in enumerate(zip(all_select_seg_idx, all_segments)):
        for layer_i, (layer_seg_i, layer_seg) in enumerate(zip(segment_idx, segments)):
            try:
                selected_segments_all[layer_i] += list(np.array(layer_seg)[layer_seg_i])
                selected_segments_all_idx[layer_i] += [(p_i, seg_i) for seg_i in layer_seg_i]
            except Exception as e:
                print(e)
                print(layer_seg)
                continue
            
    return selected_segments_all, selected_segments_all_idx
