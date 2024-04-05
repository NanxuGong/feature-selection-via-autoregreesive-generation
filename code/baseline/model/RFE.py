# Recursive Feature Elimination
# Recursive Feature Elimination (RFE).
# RFE [29] selects features by recursively selecting smaller and smaller feature subsets.
# Firstly, the predictor is trained by all features and the importance of each feature are scored by the predictor.
# After that, the least important features are deselected.
# This procedure process recursively until the desired number of features are selected.
# In the experiments, we set the selected feature number half of the feature space
import pickle

import pandas
import sklearn.ensemble
import torch
from sklearn.feature_selection import RFE
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from feature_env import FeatureEvaluator, base_path
from utils.logger import info


def gen_rfe(fe:FeatureEvaluator, k):
    results = []
    size = fe.ds_size
    x = fe.train.iloc[:, :-1]
    y = fe.train.iloc[:, -1]
    if fe.task_type == 'reg':
        # estimator = SVR(kernel="linear")
        estimator = RandomForestRegressor(random_state=0, n_jobs=128)
    else:
        estimator = RandomForestClassifier(random_state=0, n_jobs=128)
        # estimator = RandomForestClassifier(max_depth=7, random_state=0, n_jobs=128)
    selector = RFE(estimator, n_features_to_select=k, step=1)
    selector.fit(x, y)
    choice = torch.FloatTensor(selector.get_support())
    test_result = fe.report_performance(choice, flag='test', store=False)

    result = fe.report_performance(choice, flag='train', rp=False)
    info("The optimal accuracy is: {}, the optimal selection for rfe is:{}".format(test_result, choice))
    return result, choice


if __name__ == '__main__':
    task_name = 'coil-20'
    fe = FeatureEvaluator(task_name)
    load_file = f'{base_path}/history/{task_name}/fe.pkl'
    with open(load_file, 'rb') as f:
        fea_eval = pickle.load(f)
    k = -1
    p = -1
    for r in fea_eval.records.r_list:
        k_ = r.operation.sum()
        p_ = r.performance
        if p_ > p:
            p = p_
            k = int(k_)
    result, choice = gen_rfe(fe, k)
    # fe.generate_data(choice, flag='test').to_hdf(f'{base_path}/history/{task_name}.hdf', key='rfs_test')
    # fe.generate_data(choice, flag='train').to_hdf(f'{base_path}/history/{task_name}.hdf', key='rfs_train')
    # df_test = pandas.read_hdf(f'{base_path}/history/{task_name}.hdf', key='rfs_test')
    baseline_name = [
        'kbest',
        'mrmr',
        'lasso',
        'rfe',
        # 'gfs',
        'lassonet',
        'sarlfs',
        'marlfs',
    ]
    state = dict()
    state['raw'] = pandas.read_hdf(f'{base_path}/history/{task_name}.hdf', key=f'raw_test').shape[1]
    for method in baseline_name:
        state[method] = pandas.read_hdf(f'{base_path}/history/{task_name}.hdf', key=f'{method}_test').shape[1]
    # state['ours'] = pandas.read_hdf(f'{base_path}/history/{task_name}.hdf', key=f'our_test').shape[1]
    print(state)
    # info(len(fe))
    # task_name = 'openml_586'
    # fe = FeatureEvaluator(task_name)
    # gen_rfe(fe)