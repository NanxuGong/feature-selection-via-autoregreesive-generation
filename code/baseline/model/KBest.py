# K-Best Selection.
# The K-Best Selection [3] firstly ranks features by their χ2 scores with the target vector (label vector),
# and then selects the K highest scoring features.
# In the experiments, we make K equal to the number of selected features in MARLFS.
import numpy.random
import torch
from sklearn.feature_selection import SelectKBest, f_regression, f_classif

from feature_env import FeatureEvaluator
from utils.logger import info


def gen_kbest(fe: FeatureEvaluator, k):
    if fe.task_type == 'reg':
        score_func = f_regression
    else:
        score_func = f_classif
    x = fe.train.iloc[:, :-1]
    y = fe.train.iloc[:, -1]
    skb = SelectKBest(score_func=score_func, k=k)
    skb.fit(x, y)
    choice = torch.FloatTensor(skb.get_support())
    # info(f'current selection is {choice}')
    result = fe.report_performance(choice, flag='train')
    test_result = fe.report_performance(choice, flag='test', store=False)
    info("The optimal accuracy is: {}, the optimal selection for K-BEST is:{}".format(test_result, choice))
    return result, choice

if __name__ == '__main__':
    task_name = 'openml_618'
    fe = FeatureEvaluator(task_name)
    gen_kbest(fe, 10)
