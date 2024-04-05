import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
# sys.path.append('/home/xiaomeng/jupyter_base/AutoFS/code')
sys.path.append('/root/NIPSAutoFS/code/baseline/model')
print(sys.path)
from feature_env import FeatureEvaluator, base_path
from model import *
import MARLFS
from utils.logger import info
import pickle
import numpy as np
import torch
import pandas as pd
import warnings
from sklearn.feature_selection import mutual_info_regression
from lap_score import lap_score
from construct_W import construct_W

warnings.filterwarnings('ignore')

baseline_name = {
    'kbest': gen_kbest,
    'mrmr': gen_mrmr,
    'lasso': gen_lasso,
    'rfe': gen_rfe,
    # 'gfs': gen_gfs,
    'mcdm': gen_mcdm,
    'sarlfs': gen_sarlfs,
    'lassonet': gen_lassonet
}


def gen_redundancy_matrix(fe_, task_name_, choice):
    # choice: 'mutual_information', 'covariance_matrix'
    redundancy_matrix = np.zeros((fe_.original.shape[1]-1, fe_.original.shape[1]-1))
    if choice == 'mutual_information':
        for i in range(redundancy_matrix.shape[1]-1):
            for j in range(redundancy_matrix.shape[1]-1):

                mi = mutual_info_regression(fe_.original.iloc[:,i].values.reshape(-1,1), fe_.original.iloc[:,j].values.reshape(-1,1))
                redundancy_matrix[i, j] = mi
            
    elif choice == 'covariance_matrix':
        data = fe_.original.iloc[:,:-1]
        redundancy_matrix = np.cov(data, rowvar=False)
        redundancy_matrix = np.abs(redundancy_matrix)
    
    elif choice == 'pearson_correlation':
        data = fe_.original.iloc[:,:-1]
        redundancy_matrix = np.corrcoef(data, rowvar=False)
        redundancy_matrix = np.abs(redundancy_matrix)

    else:
        ValueError('Choice is wrong!')
    rm = pd.DataFrame(redundancy_matrix)
    rm.to_hdf(f'{base_path}/history/{task_name_}.hdf', key='redundancy_matrix')  
    return rm  



def gen_auto_feature_selection(fe_, task_name_, choice = 'mutual_information'):
    # calculate redundancy of each two features
    info(f'task name: {task_name_}, choice: {choice}, unsupervised: {fe_.unsupervised}')
    rm = gen_redundancy_matrix(fe_, task_name_, choice = choice)
    fe_.redundancy_matrix = rm
    
    # fe_.original.to_csv('data.csv')
    if fe_.unsupervised:
        kwargs_W = {}
        feature_np = np.array(fe_.original.iloc[:,:-1]).astype(np.float64)
        W = construct_W(feature_np, **kwargs_W)
        score = lap_score(feature_np, W=W)
        fe_.lap_score = 1 - score
        info('lap_score is calculated')

    fe_.train.to_hdf(f'{base_path}/history/{task_name_}.hdf', key='raw_train')
    fe_.test.to_hdf(f'{base_path}/history/{task_name_}.hdf', key='raw_test')
    max_accuracy, optimal_set, k = MARLFS.gen_marlfs(fe_, N_ACTIONS=2, N_STATES=64, EXPLORE_STEPS=300)
    best_train = fe_.generate_data(optimal_set, 'train')
    best_test = fe_.generate_data(optimal_set, 'test')
    best_train.to_hdf(f'{base_path}/history/{task_name_}.hdf', key='marlfs_train')
    best_test.to_hdf(f'{base_path}/history/{task_name_}.hdf', key='marlfs_test')
    pd.DataFrame(optimal_set).to_hdf(f'{base_path}/history/{task_name_}.hdf', key='marlfs_choice')
    report_head = 'marlfs'
    report_num = f'{best_test.shape[1]-1}\t'
    report_acc = '{:.4f}\t'.format(fe_.get_performance(best_test))
    for name_, func in baseline_name.items():
        p_, optimal_set = func(fe_, k)
        report_head += f'{name_}\t'
        best_train = fe_.generate_data(optimal_set,  flag='train')
        best_test = fe_.generate_data(optimal_set, flag='test')
        report_num += f'{best_test.shape[1]-1}\t'
        report_acc += '{:.4f}\t'.format(fe_.get_performance(best_test))
        best_train.to_hdf(f'{base_path}/history/{task_name_}.hdf', key=f'{name_}_train')
        best_test.to_hdf(f'{base_path}/history/{task_name_}.hdf', key=f'{name_}_test')
        pd.DataFrame(optimal_set).to_hdf(f'{base_path}/history/{task_name_}.hdf', key=f'{name_}_choice')
    print(report_head)
    print(report_num)
    print(report_acc)
    return fe_


def process(task_name_, choice, unsupervised = False):
    fea_eval = FeatureEvaluator(task_name_, unsupervised = unsupervised)
    fea_eval = gen_auto_feature_selection(fea_eval, task_name_,choice)
    
    with open(f'{base_path}/history/{task_name_}/fe.pkl', 'wb') as f:
        pickle.dump(fea_eval, f)
        
import argparse
parser = argparse.ArgumentParser(description='PyTorch Experiment')
parser.add_argument('--name', type=str, default='svmguide3')
parser.add_argument('--choice', type=str, default='mutual_information')
parser.add_argument('--unsupervised', type=str, default='False')
args, _ = parser.parse_known_args()

unsupervised_bool = False if args.unsupervised == 'False' else True
print(unsupervised_bool)
process(args.name, args.choice, unsupervised_bool)
