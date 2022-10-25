import numpy as np
import pickle as pkl
from argparse import ArgumentParser
from sklearn.preprocessing import minmax_scale
import os
from glob import glob
import tslearn.metrics as metrics
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
__tmat_threshold = 1e-2

def reshape_data(data):
    feature_num = len(data[0][0])
    data_rs = [[] for _ in range(feature_num)]
    
    for p_x in data:
        p_x = np.array(p_x)
        for i in range(feature_num):
            data_rs[i].append(p_x[:, i])
    
    return data_rs

def init_dist(dist_name):
    if dist_name == 'softdtw':
        dists = metrics.soft_dtw # softdtw
    elif dist_name == 'dtw':
        dists = metrics.dtw
    
    return dists

def cal_shapelet_distance(time_series, offset, seg_length, shapeles, dist):
    print('#'*20, 'Start building shapelet distance matrix', '#'*20)
    shapeles = np.array(shapeles)
    shapelet_distance = []
    
    for p_series in tqdm(time_series, desc='cal_shapelet_distance'):
        shapelet_distance.append([])
        p_series = np.array(p_series)[offset:]
        # 计算 seg_num
        seg_num = int(len(p_series) / seg_length)
        p_series = p_series[: seg_num * seg_length].reshape(seg_num, seg_length)
        for seg in p_series:
            shapelet_distance[-1].append([dist(seg, shapelet, gamma=0) for shapelet in shapeles])
    
    print('#'*20, 'End building shapelet distance matrix', '#'*20)
    return shapelet_distance

def build_transition_matrix(sdist, shapelet_num, percentile=None, threshold=None):
    print('#'*20, 'Start building transition matrix', '#'*20)
    sdist_flatten = [ss for s in sdist for ss in s]
    if percentile is not None:
        dist_threshold = np.percentile(sdist_flatten, percentile)
        print('threshold({}) {}, mean {}'.format(percentile, dist_threshold, np.mean(sdist_flatten)))
    else:
        dist_threshold = threshold
        print('threshold {}, mean {}'.format(dist_threshold, np.mean(sdist_flatten)))

    
    tmat = np.zeros((shapelet_num, shapelet_num), dtype=np.float32)
    n_edges = 0
    for p_sdist in tqdm(sdist, desc='build_transition_matrix'):
        p_sdist = np.array(p_sdist)
        seg_num = len(p_sdist)
        for sidx in range(seg_num - 1):
            src_dist = p_sdist[sidx, :]
            dst_dist = p_sdist[sidx + 1, :]
            src_idx = np.argwhere(src_dist <= dist_threshold).reshape(-1)
            dst_idx = np.argwhere(dst_dist <= dist_threshold).reshape(-1)
            if len(src_idx) == 0 or len(dst_idx) == 0:
                continue
            n_edges += len(src_idx) * len(dst_idx)
            src_dist[src_idx] = 1.0 - minmax_scale(src_dist[src_idx])
            dst_dist[dst_idx] = 1.0 - minmax_scale(dst_dist[dst_idx])
            for src in src_idx:
                tmat[src, dst_idx] += (src_dist[src] * dst_dist[dst_idx])
    tmat[tmat <= __tmat_threshold] = 0.0
    print('edge num: {}'.format(n_edges))
    print('#'*20, 'End building transition matrix', '#'*20)
    return tmat, sdist, dist_threshold

def gen_shapelets(shapeles_info, pdid, time_series):
    shapelets = []
    for info in shapeles_info:
        index = pdid.index(info['pdid'])
        s = time_series[index][info['sta']: info['end']]
        shapelets.append(s)
    
    return np.array(shapelets)
    
    
def gen_shapelets_process(selected_feature, data, data_all, pdid, dist, prefix, args):
    time_series = data[selected_feature]
    
    shapelet_path = glob(os.path.join(args.data_path, 'kfold-'+str(args.fold), str(selected_feature)+'*_layer'+str(args.layer_num)+'.pkl'))[0]
    shapeles_info = pkl.load(open(shapelet_path, 'rb'))
    
    
    shapeles = gen_shapelets(shapeles_info, pdid, time_series) # shapelet_num x seg_length
    seg_length = shapeles.shape[1]
    shapelet_num = shapeles.shape[0]
    save_child_dir = os.path.join(args.save_dir,
                                  prefix,
                                  'layer-'+str(args.layer_num),
                                  'offset-'+str(args.offset)+
                                  '_segL-'+str(seg_length)+
                                  '_dist-'+args.dist+
                                  '_percentile-'+str(args.percentile),
                                  'kfold-'+str(args.fold)
                                  )
    if not os.path.exists(save_child_dir):
        os.makedirs(save_child_dir)
    
    time_series = data[selected_feature]
    time_series_all = data_all[selected_feature]
    shapeles = gen_shapelets(shapeles_info, pdid, time_series) # shapelet_num x seg_length
    seg_length = shapeles.shape[1]
    shapelet_num = shapeles.shape[0]
    
    tmat_save_path = os.path.join(save_child_dir, 'feature-'+str(selected_feature)+
                             '_tmat.pkl')
    sdist_save_path = os.path.join(save_child_dir, 'feature-'+str(selected_feature)+
                             '_sdist.pkl')
    sdist_all_save_path = os.path.join(save_child_dir, 'feature-'+str(selected_feature)+
                             '_sdist_all.pkl')
    shapelet_save_path = os.path.join(save_child_dir, 'feature-'+str(selected_feature)+
                             '_shapelet.pkl')
    pkl.dump(shapeles, open(shapelet_save_path, 'wb'))
    sdist = cal_shapelet_distance(time_series, args.offset, seg_length, shapeles, dist) # patient_num x segment_num x shapelet_num
    pkl.dump(sdist, open(sdist_save_path, 'wb'))
    sdist_all = cal_shapelet_distance(time_series_all, args.offset, seg_length, shapeles, dist) # patient_num x segment_num x shapelet_num
    pkl.dump(sdist_all, open(sdist_all_save_path, 'wb'))
    tmat, trans_sdist, dist_threshold = build_transition_matrix(sdist, shapelet_num, percentile=args.percentile, threshold=None) # shapelet_num x shapelet_num
    pkl.dump(tmat, open(tmat_save_path, 'wb'))
    
    
def main(args):
    pdid = pkl.load(open(os.path.join(args.data_path, 'kfold-'+str(args.fold), 'pdid_train'), 'rb')).tolist()
    data = pkl.load(open(os.path.join(args.data_path, 'kfold-'+str(args.fold), 'x_train'), 'rb'))
    data_all = pkl.load(open(os.path.join(args.ori_data_path, 'x'), 'rb'))
    prefix = os.path.basename(args.data_path)
    
    data = reshape_data(data) # feature_num x patient_num x record_num
    data_all = reshape_data(data_all) # feature_num x patient_num x record_num
    dist = init_dist(args.dist)
    selected_feature = list(range(len(data)))
    pool = Pool(processes=len(selected_feature))
    partial_work = partial(gen_shapelets_process, data=data, data_all=data_all,pdid=pdid, dist=dist, prefix=prefix, args=args)
    results = pool.map(partial_work, selected_feature)
    
    return

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dist', default='softdtw', type=str)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--layer_num', default=2, type=int)
    parser.add_argument('--soft_dtw_gamma', default=0.5, type=float)
    parser.add_argument('--offset', default=0, type=int)
    parser.add_argument('--data_path', default='/home/jxf/code/Shapelet_GNN/logs/motality/20220911-01_Multi_ConvTransformer_clstoken_dropout5_lr5_kernel-5_auroc', type=str)
    parser.add_argument('--ori_data_path', default='/home/jxf/code/Shapelet_GNN/input/ckd_shapelet_filter_0908', type=str)
    parser.add_argument('--save_dir', default='/home/jxf/code/Shapelet_GNN/input/graph_data', type=str)
    parser.add_argument('--percentile', default=10, type=float)
    args = parser.parse_args()
    main(args)
