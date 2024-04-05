# Lasso for feature selection
# Regression shrinkage and selection via the lasso
import torch
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, LinearSVR

from sklearn.feature_selection import SelectFromModel

from feature_env import FeatureEvaluator
from utils.logger import info

from sklearn.model_selection import train_test_split
import pandas as pd


def gen_lasso(fe: FeatureEvaluator, k):
    results = []
    size = fe.ds_size
    # sklearn.model_selection.

    x = fe.train.iloc[:, :-1]
    y = fe.train.iloc[:, -1]
    if fe.task_type == 'reg':
        score_func = LinearSVR(C=1.0)
    else:
        score_func = LinearSVC(C=1.0, penalty='l1', dual=False)

    score_func.fit(x, y)
    model = SelectFromModel(score_func, prefit=True, max_features=k)
    choice = torch.FloatTensor(model.get_support())
    result = fe.report_performance(choice, flag='train', rp=False)
    test_result = fe.report_performance(choice, flag='test', store=False)
    info("The optimal accuracy is: {}, the optimal selection for LASSO is:{}".format(test_result, choice))
    return result, choice


if __name__ == '__main__':
    task_name = 'openml_589'
    fe = FeatureEvaluator(task_name)
    gen_lasso(fe, 10)
    # info(len(fe))
    # task_name = 'openml_586'
    # fe = FeatureEvaluator(task_name)
    # gen_lasso(fe)
    # info(len(fe))
