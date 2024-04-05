# Genetic Feature Selection
# Genetic algorithms in feature selection R.leardi 1996

# from sklearn.feature_selection import GenericUnivariateSelect
import pickle
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('/root/NIPSAutoFS/code/baseline')
import torch
from genetic_selection import GeneticSelectionCV
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from feature_env import FeatureEvaluator, base_path
from utils.logger import info
from sklearn.svm import LinearSVC, SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def gen_gfs(fe: FeatureEvaluator, k):
    results = []
    x = fe.train.iloc[:, :-1]
    y = fe.train.iloc[:, -1]
    if fe.task_type == 'reg':
        estimator = SVR(kernel="linear")
        # estimator = DecisionTreeRegressor()
    else:
        # estimator = SVC(kernel="linear")
        estimator = DecisionTreeClassifier()
    selector = GeneticSelectionCV(
        estimator,
        n_jobs=128,
        max_features=k,
        crossover_proba=0.5,
        mutation_proba=0.2,
        n_generations=40,cv=5,
        crossover_independent_proba=0.5,
        mutation_independent_proba=0.05
    )
    selector = selector.fit(x, y)
    choice = torch.FloatTensor(selector.get_support())
    # info(f'current selection is {choice}')
    result = fe.report_performance(choice, flag='train')
    test_result = fe.report_performance(choice, flag='test', store=False)
    info("The optimal accuracy is: {}, the optimal selection for GFS is:{}".format(test_result, choice))
    return test_result, choice

if __name__ == '__main__':
    task_name = 'coil-20'
    results = []
    # for task_name in ['spectf', 'svmguide3', 'german_credit', 'uci_credit_card', 'spam_base',
    #                'megawatt1','ionosphere', 'activity', 'mice_protein','coil-20', 'minist', 'minist_fashion' ,'openml_586', 'openml_589', 'openml_607', 'openml_616',
    #                 'openml_618', 'openml_620',
    #               'openml_637']:
    for task_name in ['openml_618']:
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
        result, choice = gen_gfs(fe, k)
        results.append((task_name, result))
        fe.generate_data(choice, 'train').to_hdf(f'{base_path}/history/{task_name}.hdf', 'gfs_train')
        fe.generate_data(choice, 'test').to_hdf(f'{base_path}/history/{task_name}.hdf', 'gfs_test')
    print(results)
# [('mice_protein', 0.7734672304439747), ('coil-20', 0.9548699334543255),('openml_589', 0.4472012360175109), ('openml_607', 0.45701822454188434)]