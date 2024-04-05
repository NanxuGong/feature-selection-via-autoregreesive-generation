import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('/home/xiaomeng/jupyter_base/AutoFS/code')
print(sys.path)
from feature_env import FeatureEvaluator, base_path
from baseline.model import *
import pickle

import warnings

warnings.filterwarnings('ignore')

task = ['ap_omentum_ovary']

baseline_name = {
    'kbest': gen_kbest,
    'mrmr': gen_mrmr,
    'lasso': gen_lasso,
    'rfe': gen_rfe,
    'gfs': gen_gfs,
    # 'sarlfs': gen_sarlfs,  # feature_env, N_STATES, N_ACTIONS, EPISODE=-1, EXPLORE_STEPS=30
    # 'marlfs': gen_marlfs,  # feature_env, N_STATES, N_ACTIONS, EPISODE=-1, EXPLORE_STEPS=30
    'lassonet': gen_lassonet
}


def gen_auto_feature_selection(fe_, task_name_):
    fe_.train.to_hdf(f'{base_path}/history/{task_name_}_o.hdf', key='raw_train')
    fe_.test.to_hdf(f'{base_path}/history/{task_name_}_o.hdf', key='raw_test')
    max_accuracy, optimal_set, k = gen_sarlfs_init(fe_, EXPLORE_STEPS=300)
    best_train = fe_.generate_data(optimal_set, 'train')
    best_test = fe_.generate_data(optimal_set, 'test')
    best_train.to_hdf(f'{base_path}/history/{task_name_}_o.hdf', key='sarlfs_train')
    best_test.to_hdf(f'{base_path}/history/{task_name_}_o.hdf', key='sarlfs_test')
    for name_, func in baseline_name.items():
        p_, optimal_set = func(fe_, k)
        best_train = fe_.generate_data(optimal_set, flag='train')
        best_test = fe_.generate_data(optimal_set, flag='test')
        best_train.to_hdf(f'{base_path}/history/{task_name_}_o.hdf', key=f'{name_}_train')
        best_test.to_hdf(f'{base_path}/history/{task_name_}_o.hdf', key=f'{name_}_test')
    return k


def process(task_name_):
    fea_eval = FeatureEvaluator(task_name_)
    gen_auto_feature_selection(fea_eval, task_name_)
    with open(f'{base_path}/history/{task_name_}/fe_o.pkl', 'wb') as f:
        pickle.dump(fea_eval, f)


import argparse

parser = argparse.ArgumentParser(description='PyTorch Experiment')
parser.add_argument('--name', type=str)
args, _ = parser.parse_known_args()

process(args.name)


# ' nohup /home/xiaomeng/miniconda3/envs/shaow/bin/python -u /home/xiaomeng/jupyter_base/AutoFS/code/baseline/automatic_feature_selection_gen.py --name spectf > /home/xiaomeng/jupyter_base/AutoFS/code/baseline/script/spectf.log &'

'nohup /home/xiaomeng/miniconda3/envs/shaow/bin/python -u /home/xiaomeng/jupyter_base/AutoFS/code/baseline/automatic_fs_with_huge_ds.py --name ap_omentum_ovary > /home/xiaomeng/jupyter_base/AutoFS/code/baseline/script/ap_omentum_ovary.log & '