# filter methods
# mRMR. The mRMR [28] firstly ranks features by minimizing feature’s redundancy,
# while maximizing their relevance with the target vector (label vector), and then
# selects the K highest ranking features.
# In the experiments, we make K equal to the number of selected features in MARLFS.
from mrmr import mrmr_classif, mrmr_regression
from utils.logger import info
import torch
from feature_env import FeatureEvaluator


def gen_mrmr(fe:FeatureEvaluator, k):
    results = []
    size = fe.ds_size
    x = fe.train.iloc[:, :-1]
    y = fe.train.iloc[:, -1]
    choice = torch.zeros(fe.ds_size)
    if fe.task_type == 'reg':
        choice_indice = torch.LongTensor(mrmr_regression(x, y, K=k, show_progress=False, n_jobs=128))
    else:
        choice_indice = torch.LongTensor(mrmr_classif(x, y, K=k, show_progress=False, n_jobs=128))
    choice[choice_indice] = 1.
    # info(f'current selection is {choice}')
    test_result = fe.report_performance(choice, flag='test', store=False)
    result = fe.report_performance(choice, flag='train', rp=False)
    info("The optimal accuracy is: {}, the optimal selection for mRMR is:{}".format(test_result, choice))
    return result, choice


if __name__ == '__main__':
    task_name = 'higgs'
    fe = FeatureEvaluator(task_name)
    gen_mrmr(fe, 10)
    info(len(fe))
    task_name = 'openml_586'
    fe = FeatureEvaluator(task_name)
    gen_mrmr(fe, 10)