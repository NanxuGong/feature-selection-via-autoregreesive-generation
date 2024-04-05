import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('/home/xiaomeng/jupyter_base/AutoFS/code')
print(sys.path)
from feature_env import FeatureEvaluator, base_path
import MARLFS
from utils.logger import info
import pickle

import warnings

warnings.filterwarnings('ignore')

baseline_name = {
    # 'kbest': gen_kbest,
    # 'mrmr': gen_mrmr,
    # 'lasso': gen_lasso,
    # 'rfe': gen_rfe,
    # 'gfs': gen_gfs,
    # 'sarlfs': gen_sarlfs,  # feature_env, N_STATES, N_ACTIONS, EPISODE=-1, EXPLORE_STEPS=30
    'marlfs': MARLFS.gen_marlfs  # feature_env, N_STATES, N_ACTIONS, EPISODE=-1, EXPLORE_STEPS=30
    # 'lassonet':gen_lassonet
}

# task = ['spectf', 'svmguide3', 'german_credit', 'spam_base',
#                   'ionosphere', 'megawatt1', 'uci_credit_card',
#                   'openml_618', 'openml_589', 'openml_616', 'openml_607', 'openml_620',
#                   'openml_637',
#                   'openml_586', 'higgs']
task = [
                  'openml_618', 'openml_589', 'openml_616', 'openml_607', 'openml_620',
                  'openml_637',
                  'openml_586', 'higgs']
def gen_auto_feature_selection(fe_:FeatureEvaluator, task_name_):
    report_ = []
    # max_accuracy, optimal_set, k = gen_marlfs(fe_, N_ACTIONS=2, N_STATES=64, EXPLORE_STEPS=300)
    # best_set = fe_.generate_data(optimal_set)
    # best_set.to_hdf(f'{base_path}/history/{task_name_}/best-marlfs.hdf', key='xm')
    # report_.append(('marlfs', max_accuracy))
    k = -1
    p = -1
    for r in fe_.records.r_list:
        k_ = r.operation.sum()
        p_ = r.performance
        if p_ > p :
            p = p_
            k = int(k_)
    info(f'the k for task : {task_name_} is {k}')
    for name_, method in baseline_name.items():
        func = baseline_name[name_]
        p_, optimal_set = func(fe_, k)
        info(f'done task for {task_name_} with method {name_} with performance {p_}')
        best_set = fe_.generate_data(optimal_set)
        best_set.to_hdf(f'{base_path}/history/{task_name_}/best-{name_}.hdf', key='wdj')
        info(f'save to {base_path}/history/{task_name_}/best-{name_}.hdf')
        report_.append((name_, p_))
    fe_.save()
    return report_, k


def process(task_name_):
    load_file = f'{base_path}/history/{task_name_}/fe.pkl'
    with open(load_file, 'rb') as f:
        fea_eval = pickle.load(f)
    report, k = gen_auto_feature_selection(fea_eval, task_name_)

    with open(f'{base_path}/history/{task_name_}/fe+.pkl', 'wb') as f:
        pickle.dump(fea_eval, f)

    print(report)
    with open(f'{base_path}/baseline_log_{task_name_}+.log', 'w') as f:
        strings = f'{task_name_} : {len(fea_eval)} : k {k} =>  '
        for name, p in report:
            strings += f'{name} : {p}  '
        f.writelines(strings)


import argparse

parser = argparse.ArgumentParser(description='PyTorch Experiment')
parser.add_argument('--name', type=str)
args, _ = parser.parse_known_args()
process('uci_credit_card')
