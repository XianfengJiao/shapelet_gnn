import os
import pickle as pkl
from glob import glob

def load_record(record_path):
    record_dict = {}
    for record_fp in glob(os.path.join(record_path, '*_sdist.pkl')):
        i = os.path.basename(record_fp).split('-')[1].split('_')[0]
        record_dict['f'+i] = pkl.load(open(record_fp, 'rb'))
    
    record_label = pkl.load(open(os.path.join(record_path, 'y.pkl'), 'rb'))
    record_label = [yy[-1] for yy in record_label]
    return record_dict, record_label