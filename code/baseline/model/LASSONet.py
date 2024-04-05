import torch
from lassonet import LassoNetClassifierCV, LassoNetRegressorCV
from sklearn import preprocessing

from feature_env import FeatureEvaluator
from utils.logger import info


def gen_lassonet(fe:FeatureEvaluator, k):
    results = []
    x = fe.train.iloc[:, :-1].to_numpy()
    y = fe.train.iloc[:, -1].to_numpy()
    if fe.task_type == 'reg':
        selector = LassoNetRegressorCV()
    else:
        normalizer = preprocessing.Normalizer()
        normalizer.fit(x)
        x = normalizer.transform(x)
        selector = LassoNetClassifierCV()  # LassoNetRegressorCV
    selector = selector.fit(x, y)
    scores = selector.feature_importances_
    value, indice = torch.topk(scores, k)
    choice = torch.zeros(x.shape[1])
    choice[indice] = 1
    result = fe.report_performance(choice, flag='train')
    test_result = fe.report_performance(choice, flag='test', store=False)
    info("The optimal accuracy is: {}, the optimal selection for LASSONET is:{}".format(test_result, choice))
    return result, choice

if __name__ == '__main__':
    task_name = 'openml_637'
    fe = FeatureEvaluator(task_name)
    gen_lassonet(fe, 28)



