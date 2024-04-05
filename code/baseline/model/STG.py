import sklearn.ensemble
import torch

from feature_env import FeatureEvaluator
from utils.logger import info

from stg import STG


def gen_stg(fe:FeatureEvaluator, k, device=7):
    results = []
    size = fe.ds_size
    x = fe.train.iloc[:, :-1].to_numpy()
    y = fe.train.iloc[:, -1].to_numpy()
    device = torch.device(f'cuda:{device}')
    if fe.task_type == 'reg':
        selector = STG(task_type='regression',input_dim=x.shape[1], output_dim=1, hidden_dims=[64, 128, 64],
                       activation='tanh', optimizer='SGD', learning_rate=0.0001, batch_size=x.shape[0],
                       feature_selection=True, sigma=0.5, lam=0.1, random_state=1, device=device)
    else:
        selector = STG(task_type='classification',input_dim=x.shape[1], output_dim=1, hidden_dims=[64, 128, 64],
                       activation='tanh', optimizer='SGD', learning_rate=0.0001, batch_size=x.shape[0],
                       feature_selection=True, sigma=0.5, lam=0.1, random_state=1, device=device)


    selector.fit(x, y, nr_epochs=3000, valid_X=x, valid_y=y, print_interval=1000)
    scores = torch.tensor(selector.get_gates('raw'))
    value, indice = torch.topk(scores, k)
    choice = torch.zeros(x.shape[1])
    choice[indice] = 1
    info(f'current selection is {choice}')
    result = fe.report_performance(choice, flag='test')
    info("The optimal accuracy is: {}, the optimal selection for each feature is:{}".format(result, choice))
    return result, choice


if __name__ == '__main__':
    task_name = 'openml_586'
    fe = FeatureEvaluator(task_name)
    gen_stg(fe, 10)



