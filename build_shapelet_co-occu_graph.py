import enum
import numpy as np
import pickle as pkl
from argparse import ArgumentParser
from sklearn.preprocessing import minmax_scale
import os
import tslearn.metrics as metrics
from tqdm import tqdm
from glob import glob
__co_mat_threshold = 1e-2

def load_data(sdist_dir):
    sdist_paths = glob(os.path.join(sdist_dir, 'feature-*_sdist.pkl'))
    feature_num = len(sdist_paths)
    sdists = [None for _ in range(feature_num)]
    
    for sdist_p in sdist_paths:
        feature_id = int(sdist_p.split('/')[-1].split('_')[0].split('-')[-1])
        sdists[feature_id] = pkl.load(open(sdist_p, 'rb'))
    
    return sdists

def reshape_data(data):
    patient_num = len(data[0])
    data_r = [[] for _ in range(patient_num)]
    
    for f_i, f_data in enumerate(data):
        for p_i, p_data in enumerate(f_data):
            for seg_i, seg_data in enumerate(p_data):
                if f_i == 0:
                    data_r[p_i].append([])
                data_r[p_i][seg_i].append(seg_data)
    return data_r

def build_co_occurence_matrix(sdists, feature_num, shapelet_num, thresholds):
    print('#'*20, 'Start building co-occurence matrix', '#'*20)
    
    co_mat = np.zeros((shapelet_num * feature_num, shapelet_num * feature_num), dtype=np.float32)
    n_edges = 0
    for p_sdist in tqdm(sdists):
        p_sdist = np.array(p_sdist)
        for seg_sdist in p_sdist:
            f_num = len(seg_sdist)
            # 遍历 feature ,建立共现关系矩阵
            for f_src_i in range(f_num - 1):
                for f_dst_i in range(f_src_i + 1, f_num):
                    src_dist = seg_sdist[f_src_i, :]
                    dst_dist = seg_sdist[f_dst_i, :]
                    src_idx = np.argwhere(src_dist <= thresholds[f_src_i]).reshape(-1)
                    dst_idx = np.argwhere(dst_dist <= thresholds[f_dst_i]).reshape(-1)
                    if len(src_idx) == 0 or len(dst_idx) == 0:
                        continue
                    n_edges += len(src_idx) * len(dst_idx) * 2
                    src_dist[src_idx] = 1.0 - minmax_scale(src_dist[src_idx])
                    dst_dist[dst_idx] = 1.0 - minmax_scale(dst_dist[dst_idx])
                    for src in src_idx:
                        co_mat[f_src_i * shapelet_num + src, f_dst_i * shapelet_num + dst_idx] += (src_dist[src] * dst_dist[dst_idx])
                        co_mat[f_dst_i * shapelet_num + dst_idx, f_src_i * shapelet_num + src] += (src_dist[src] * dst_dist[dst_idx])
                        
    co_mat[co_mat <= __co_mat_threshold] = 0.0
    print('edge num: {}'.format(n_edges))
    
    
    print('#'*20, 'End building co-occurence matrix', '#'*20)
    return co_mat, sdists

def cal_thresholds(sdists, percentile):
    thresholds = []
    
    for f_sdist in sdists:
        sdist_flatten = [ss for s in f_sdist for ss in s]
        thr = np.percentile(sdist_flatten, percentile)
        thresholds.append(thr)
    
    return thresholds

def main(args):
    
    sdists = load_data(args.sdist_dir) # feature_num x patient_num x segment_num x shapelet_num
    thresholds = cal_thresholds(sdists, args.percentile)
    sdists = reshape_data(sdists) # patient_num x segment_num x feature_num x shapelet_num
    
    feature_num = len(sdists[0][0])
    shapelet_num = len(sdists[0][0][0])
    
    co_mat, _ = build_co_occurence_matrix(sdists, feature_num, shapelet_num, thresholds=thresholds) # (feature_num x shapelet_num) x (feature_num x shapelet_num)
    pkl.dump(co_mat, open(os.path.join(args.sdist_dir, 'co_mat.pkl'), 'wb'))    
    return

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sdist_dir', default='/home/jxf/code/pytorch-lightning-med/med-seq/graph_utils/gen_graph_data/test_sdist', type=str)
    parser.add_argument('--percentile', default=5, type=float)
    args = parser.parse_args()
    main(args)
