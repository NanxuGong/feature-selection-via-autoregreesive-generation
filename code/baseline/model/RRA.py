import pickle

from utils.logger import info
import torch

from baseline.model.MCDM import rest
from feature_env import FeatureEvaluator, base_path
import pandas
from RobustRankingAggregate import rankagg


def gen_rra(fe: FeatureEvaluator, k):
    x = fe.train.iloc[:, :-1]
    y = fe.train.iloc[:, -1]
    accumulated = rest(x, y, fe.task_type)
    norm_importance = []
    for labels in accumulated:
        labels = labels.reshape(-1)
        min_val = min(labels)
        max_val = max(labels)
        train_encoder_target = [(i - min_val) / (max_val - min_val) for i in labels]
        norm_importance.append(train_encoder_target)
    importances = torch.FloatTensor(norm_importance).reshape(len(norm_importance[0]), len(norm_importance))
    order = importances.argsort(descending=True)
    score = torch.zeros_like(order, dtype=torch.float)
    for index, i in enumerate(order):
        for j, pos in zip(range(order.shape[1]), i):
            score[index, pos] = (order.shape[1] - j - 1 + 0.) / order.shape[1]
    # print(importances)
    rank = torch.argsort(torch.tensor(rankagg(pandas.DataFrame(importances.numpy())).to_numpy()).reshape(-1), descending=True)
    # print('aggre', [int(i) for i, j in rank])
    selected = rank[:k]
    # info(f'current selection is {choice}')
    choice = torch.zeros(fe.ds_size)
    choice[selected] = 1
    test_result = fe.report_performance(choice, flag='test', store=False, rp=False)
    result = fe.report_performance(choice, flag='train', rp=False)
    info("The optimal on {} accuracy is: {}".format(fe.task_name, test_result))
    info("The optimal accuracy is: {}, the optimal selection for rra is:{}".format(test_result, choice))
    return result, choice


if __name__ == '__main__':
    for task_name in [
        'spectf', 'svmguide3', 'german_credit', 'uci_credit_card', 'spam_base', 'megawatt1','ionosphere', 'activity',
                      'mice_protein', 'coil-20', 'minist', 'minist_fashion',
                      'openml_586', 'openml_589', 'openml_607', 'openml_616','openml_618','openml_620','openml_637']:
        load_file = f'{base_path}/history/{task_name}/fe.pkl'
        with open(load_file, 'rb') as f:
            fe = pickle.load(f)
        k = -1
        p = -1
        o = None
        for r in fe.records.r_list:
            k_ = r.operation.sum()
            p_ = r.performance
            if p_ > p:
                p = p_
                o = r.operation
                k = int(k_)
        info(str(k))
        k = int(fe.ds_size /2)
        info(f'the k for task : {task_name} is {k}')
        res, optimal_set = gen_rra(fe, k=k)
        best_train = fe.generate_data(optimal_set, flag='train')
        best_test = fe.generate_data(optimal_set, flag='test')
        best_train.to_hdf(f'{base_path}/history/rra_train.hdf', key=f'{task_name}')
        best_test.to_hdf(f'{base_path}/history/rra_test.hdf', key=f'{task_name}')