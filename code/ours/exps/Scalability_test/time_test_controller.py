import argparse
import os
import sys
import time

import mcdm
import pandas
import tqdm
from genetic_selection import GeneticSelectionCV
from lassonet import LassoNetRegressorCV, LassoNetClassifierCV
from mrmr import mrmr_regression, mrmr_classif
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import f_regression, f_classif, SelectKBest, RFE, SelectFromModel
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('/home/xiaomeng/jupyter_base/AutoFS/code')

from utils.tools import test_task_wo_cv
from baseline.model.MCDM import rest
from baseline.model.RobustRankingAggregate import rankagg


import pickle
import random
import sys
from typing import List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils
from torch import Tensor
from torch.utils.data import DataLoader

from ours.controller import GAFS
from feature_env import FeatureEvaluator, base_path
from ours.utils_meter import AvgrageMeter, pairwise_accuracy, hamming_distance, count_parameters_in_MB, FSDataset
from record import SelectionRecord
from utils.logger import info, error

parser = argparse.ArgumentParser()
# Basic model parameters.
# /home/xiaomeng/miniconda3/envs/shaow/bin/python3 -u /home/xiaomeng/jupyter_base/AutoFS/code/ours/exps/Scalability_test/time_test_controller.py --mlp_layers 3 --encoder_layers 2 --task_name svmguide3
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--new_gen', type=int, default=10)
parser.add_argument('--method_name', type=str, choices=['rnn'], default='rnn')
parser.add_argument('--task_name', type=str, choices=['spectf', 'svmguide3', 'german_credit', 'spam_base',
                                                      'ionosphere', 'megawatt1', 'uci_credit_card', 'openml_618',
                                                      'openml_589', 'openml_616', 'openml_607', 'openml_620',
                                                      'openml_637',
                                                      'openml_586', 'uci_credit_card', 'higgs', 'ap_omentum_ovary','activity'
                                                      , 'mice_protein', 'coil-20', 'isolet', 'minist', 'minist_fashion'], default='german_credit')
parser.add_argument('--gpu', type=int, default=7, help='used gpu')
parser.add_argument('--fe', type=str, choices=['+', '', '-'], default='-')
parser.add_argument('--top_k', type=int, default=100)
parser.add_argument('--gen_num', type=int, default=25)
parser.add_argument('--encoder_layers', type=int, default=1)
parser.add_argument('--encoder_hidden_size', type=int, default=64)
parser.add_argument('--encoder_emb_size', type=int, default=32)
parser.add_argument('--mlp_layers', type=int, default=2)
parser.add_argument('--mlp_hidden_size', type=int, default=200)
parser.add_argument('--decoder_layers', type=int, default=1)
parser.add_argument('--decoder_hidden_size', type=int, default=64)
parser.add_argument('--encoder_dropout', type=float, default=0)
parser.add_argument('--mlp_dropout', type=float, default=0)
parser.add_argument('--decoder_dropout', type=float, default=0)
parser.add_argument('--l2_reg', type=float, default=0.0)
parser.add_argument('--max_step_size', type=int, default=100)
parser.add_argument('--trade_off', type=float, default=0.8)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--grad_bound', type=float, default=5.0)

args = parser.parse_args()
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


def gafs_train(train_queue, model: GAFS, optimizer):
    objs = AvgrageMeter()
    mse = AvgrageMeter()
    nll = AvgrageMeter()
    model.train()
    for step, sample in enumerate(train_queue):
        encoder_input = sample['encoder_input']
        encoder_target = sample['encoder_target']
        decoder_input = sample['decoder_input']
        decoder_target = sample['decoder_target']

        encoder_input = encoder_input.cuda(model.gpu)
        encoder_target = encoder_target.cuda(model.gpu).requires_grad_()
        decoder_input = decoder_input.cuda(model.gpu)
        decoder_target = decoder_target.cuda(model.gpu)

        optimizer.zero_grad()
        predict_value, log_prob, arch = model.forward(encoder_input, decoder_input)
        loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze()) # mse loss
        loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1)) # ce loss
        loss = args.trade_off * loss_1 + (1 - args.trade_off) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()

        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)

    return objs.avg, mse.avg, nll.avg


def gafs_valid(queue, model: GAFS):
    pa = AvgrageMeter()
    hs = AvgrageMeter()
    mse = AvgrageMeter()
    with torch.no_grad():
        model.eval()
        for step, sample in enumerate(queue):
            encoder_input = sample['encoder_input']
            encoder_target = sample['encoder_target']
            decoder_target = sample['decoder_target']

            encoder_input = encoder_input.cuda(model.gpu)
            encoder_target = encoder_target.cuda(model.gpu)
            decoder_target = decoder_target.cuda(model.gpu)

            predict_value, logits, arch = model.forward(encoder_input)
            n = encoder_input.size(0)
            pairwise_acc = pairwise_accuracy(encoder_target.data.squeeze().tolist(),
                                             predict_value.data.squeeze().tolist())
            hamming_dis = hamming_distance(decoder_target.data.squeeze().tolist(), arch.data.squeeze().tolist())
            mse.update(F.mse_loss(predict_value.data.squeeze(), encoder_target.data.squeeze()), n)
            pa.update(pairwise_acc, n)
            hs.update(hamming_dis, n)
    return mse.avg, pa.avg, hs.avg


def choice_to_onehot(choice: List[int]):
    size = len(choice)
    onehot = torch.zeros(size + 1)
    onehot[torch.tensor(choice)] = 1
    return onehot[:-1]
    # if choice.dim() == 1:
    #     selected = torch.zeros_like(choice)
    #     selected[choice] = 1
    #     return selected[1:-1]
    # else:
    #     onehot = torch.empty_like(choice)
    #     for i in range(choice.shape[0]):
    #         onehot[i] = choice_to_onehot(choice[i])
    #     return onehot


def gafs_infer(queue, model, step, direction='+'):
    new_gen_list = []
    model.eval()
    for i, sample in enumerate(queue):
        encoder_input = sample['encoder_input']
        encoder_input = encoder_input.cuda(model.gpu)
        model.zero_grad()
        new_gen = model.generate_new_feature(encoder_input, predict_lambda=step, direction=direction)
        new_gen_list.extend(new_gen.data.squeeze().tolist())
    return new_gen_list


def select_top_k(choice: Tensor, labels: Tensor, k: int) -> (Tensor, Tensor):
    values, indices = torch.topk(labels, k, dim=0)
    return choice[indices.squeeze()], labels[indices.squeeze()]


def main():
    if not torch.cuda.is_available():
        info('No GPU found!')
        sys.exit(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
    info('test time begin!!!')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    device = int(args.gpu)
    info(f"Args = {args}")

    with open(f'{base_path}/history/{args.task_name}/fe.pkl', 'rb') as f:
        fe: FeatureEvaluator = pickle.load(f)
    model = GAFS(fe, args)
    choice, labels = fe.get_record(args.gen_num, eos=fe.ds_size)
    valid_choice, valid_labels = fe.get_record(0, eos=fe.ds_size)
    if os.path.exists(f'{base_path}/history/{fe.task_name}/GAFS.model_dict'):
        model.load_state_dict(torch.load(f'{base_path}/history/{fe.task_name}/GAFS.model_dict'))
        info(f"load the model from local...")
        model = model.cuda(device)
    else:

        model = model.cuda(device)
        info('Training Encoder-Predictor-Decoder')
        min_val = min(labels)
        max_val = max(labels)
        train_encoder_target = [(i - min_val) / (max_val - min_val) for i in labels]
        valid_encoder_target = [(i - min_val) / (max_val - min_val) for i in valid_labels]
        train_dataset = FSDataset(choice, train_encoder_target, train=True, sos_id=fe.ds_size, eos_id=fe.ds_size)
        valid_dataset = FSDataset(valid_choice, valid_encoder_target, train=False, sos_id=fe.ds_size, eos_id=fe.ds_size)
        train_queue = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        valid_queue = torch.utils.data.DataLoader(
            valid_dataset, batch_size=len(valid_dataset), shuffle=False, pin_memory=True)
        nao_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
        losses = []
        mse_losses = []
        ce_losses = []
        for _ in tqdm.tqdm(range(1, args.epochs + 1)):
            nao_loss, nao_mse, nao_ce = gafs_train(train_queue, model, nao_optimizer)
            losses.append(nao_loss)
            mse_losses.append(nao_mse)
            ce_losses.append(nao_ce)
        loss_df = pandas.DataFrame([losses, mse_losses, ce_losses])
        print(loss_df)
        loss_df.to_csv(f'{base_path}/history/{fe.task_name}/loss.csv')
    top_selection, top_performance = select_top_k(valid_choice, valid_labels, args.top_k)
    infer_dataset = FSDataset(top_selection, top_performance, False, sos_id=fe.ds_size, eos_id=fe.ds_size)
    infer_queue = DataLoader(infer_dataset, batch_size=len(infer_dataset), shuffle=False,
                             pin_memory=True)
    new_selection = []
    new_choice = []
    predict_step_size = 0
    # start = time.time()
    time_line = []
    for i in range(10):
        predict_step_size += 1
        # info('Generate new architectures with step size {:d}'.format(predict_step_size))
        start = time.time()
        new_record = gafs_infer(infer_queue, model, direction='+', step=predict_step_size)
        end = time.time()
        time_line.append(end - start)
        # for choice in new_record:
        #     onehot_choice = choice_to_onehot(choice)
        #     if onehot_choice.sum() <= 0:
        #         error('insufficient selection')
        #         continue
        #     record = SelectionRecord(onehot_choice.numpy(), -1)
        #     if record not in fe.records.r_list and record not in new_selection:
        #         new_selection.append(record)
        #         new_choice.append(onehot_choice)
        #     if len(new_selection) >= args.new_gen:
        #         break
        # info(f'{len(new_selection)} new choice generated now', )
        # if predict_step_size > args.max_step_size:
        #     break
    # info(f'build {len(new_selection)} new choice !!!')
    # end = time.time()
    info(f'the total infer time for GAINS is {np.sum(time_line)}')
    info(f'the each infer time for GAINS is {time_line}')
    info(f'the average infer time for one step is {np.mean(time_line)}')
    info(f'the total infer std for GAINS is {np.std(time_line)}')
    info(f"param size = {count_parameters_in_MB(model)}MB")
    # time_line2 = []
    # start = time.time()
    # for s in new_selection:
    #     # train_data = fe.generate_data(s.operation, 'train')
    #     # result = fe.get_performance(train_data)
    #     start = time.time()
    #     test_data = fe.generate_data(s.operation, 'test')
    #     test_task_wo_cv(test_data, fe.task_type)
    #     end = time.time()
    #     time_line2.append(end - start)
    # end = time.time()
    # info(f'the total search time for GAINS is {end - start}')
    # k = -1
    # for r in fe.records.r_list:
    #     k_ = r.operation.sum()
    #     if int(k_) > k:
    #         k = int(k_)
    # '''
    # kbest
    # '''
    # start = time.time()
    # if fe.task_type == 'reg':
    #     score_func = f_regression
    # else:
    #     score_func = f_classif
    # x = fe.train.iloc[:, :-1]
    # y = fe.train.iloc[:, -1]
    # skb = SelectKBest(score_func=score_func, k=k)
    # skb.fit(x, y)
    # choice = torch.FloatTensor(skb.get_support())
    # end = time.time()
    # info(f'the total infer time for KBEST is {end - start}')
    # '''
    # mrmr
    # '''
    # start = time.time()
    # x = fe.train.iloc[:, :-1]
    # y = fe.train.iloc[:, -1]
    # choice = torch.zeros(fe.ds_size)
    # if fe.task_type == 'reg':
    #     choice_indice = torch.LongTensor(mrmr_regression(x, y, K=k, show_progress=False, n_jobs=128))
    # else:
    #     choice_indice = torch.LongTensor(mrmr_classif(x, y, K=k, show_progress=False, n_jobs=128))
    # choice[choice_indice] = 1.
    # end = time.time()
    # info(f'the total infer time for mRMR is {end - start}')
    # '''
    # lasso
    # '''
    # start = time.time()
    # x = fe.train.iloc[:, :-1]
    # y = fe.train.iloc[:, -1]
    # if fe.task_type == 'reg':
    #     score_func = LinearSVR(C=1.0)
    # else:
    #     score_func = LinearSVC(C=1.0, penalty='l1', dual=False)
    #
    # score_func.fit(x, y)
    # model = SelectFromModel(score_func, prefit=True, max_features=k)
    # choice = torch.FloatTensor(model.get_support())
    # end = time.time()
    # info(f'the total infer time for Lasso is {end - start}')
    # '''
    # rfe
    # '''
    # start = time.time()
    # x = fe.train.iloc[:, :-1]
    # y = fe.train.iloc[:, -1]
    # if fe.task_type == 'reg':
    #     # estimator = SVR(kernel="linear")
    #     estimator = RandomForestRegressor(random_state=0, n_jobs=128)
    # else:
    #     estimator = RandomForestClassifier(random_state=0, n_jobs=128)
    #     # estimator = RandomForestClassifier(max_depth=7, random_state=0, n_jobs=128)
    # selector = RFE(estimator, n_features_to_select=k, step=1)
    # selector.fit(x, y)
    # choice = torch.FloatTensor(selector.get_support())
    # end = time.time()
    # info(f'the total infer time for RFE is {end - start}')
    # '''
    # lassonet
    # '''
    # start = time.time()
    # x = fe.train.iloc[:, :-1].to_numpy()
    # y = fe.train.iloc[:, -1].to_numpy()
    # if fe.task_type == 'reg':
    #     selector = LassoNetRegressorCV()
    # else:
    #     normalizer = preprocessing.Normalizer()
    #     normalizer.fit(x)
    #     x = normalizer.transform(x)
    #     selector = LassoNetClassifierCV()  # LassoNetRegressorCV
    # selector = selector.fit(x, y)
    # scores = selector.feature_importances_
    # value, indice = torch.topk(scores, k)
    # choice = torch.zeros(x.shape[1])
    # choice[indice] = 1
    # end = time.time()
    # info(f'the total infer time for LassoNet is {end - start}')
    # '''
    # gfs
    # '''
    # start = time.time()
    # x = fe.train.iloc[:, :-1]
    # y = fe.train.iloc[:, -1]
    # if fe.task_type == 'reg':
    #     estimator = SVR(kernel="linear")
    #     # estimator = DecisionTreeRegressor()
    # else:
    #     # estimator = SVC(kernel="linear")
    #     estimator = DecisionTreeClassifier()
    # selector = GeneticSelectionCV(
    #     estimator,
    #     n_jobs=128,
    #     max_features=k,
    #     crossover_proba=0.5,
    #     mutation_proba=0.2,
    #     n_generations=40, cv=5,
    #     crossover_independent_proba=0.5,
    #     mutation_independent_proba=0.05
    # )
    # selector = selector.fit(x, y)
    # choice = torch.FloatTensor(selector.get_support())
    # end = time.time()
    # info(f'the total infer time for GFS is {end - start}')
    # '''
    # rra
    # '''
    # accumulated = rest(x, y, fe.task_type)
    # start = time.time()
    # norm_importance = []
    # for labels in accumulated:
    #     labels = labels.reshape(-1)
    #     min_val = min(labels)
    #     max_val = max(labels)
    #     train_encoder_target = [(i - min_val) / (max_val - min_val) for i in labels]
    #     norm_importance.append(train_encoder_target)
    # importances = torch.FloatTensor(norm_importance).reshape(len(norm_importance[0]), len(norm_importance))
    # order = importances.argsort(descending=True)
    # score = torch.zeros_like(order, dtype=torch.float)
    # for index, i in enumerate(order):
    #     for j, pos in zip(range(order.shape[1]), i):
    #         score[index, pos] = (order.shape[1] - j - 1 + 0.) / order.shape[1]
    # # print(importances)
    # rank = torch.argsort(torch.tensor(rankagg(pandas.DataFrame(importances.numpy())).to_numpy()).reshape(-1),
    #                      descending=True)
    # # print('aggre', [int(i) for i, j in rank])
    # selected = rank[:k]
    # # info(f'current selection is {choice}')
    # choice = torch.zeros(fe.ds_size)
    # choice[selected] = 1
    # end = time.time()
    # info(f'the total infer time for RRA is {end - start}')
    # '''
    # mcdm
    # '''
    # start = time.time()
    # norm_importance = []
    # for labels in accumulated:
    #     labels = labels.reshape(-1)
    #     min_val = min(labels)
    #     max_val = max(labels)
    #     train_encoder_target = [(i - min_val) / (max_val - min_val) for i in labels]
    #     norm_importance.append(train_encoder_target)
    # importances = torch.FloatTensor(norm_importance).reshape(len(norm_importance[0]), len(norm_importance))
    # order = importances.argsort(descending=True)
    # score = torch.zeros_like(order, dtype=torch.float)
    # for index, i in enumerate(order):
    #     for j, pos in zip(range(order.shape[1]), i):
    #         score[index, pos] = (order.shape[1] - j - 1 + 0.) / order.shape[1]
    # alt_name = [str(i) for i in range(x.shape[1])]
    # # print(importances)
    # if fe.task_type == 'reg':
    #     rank = mcdm.rank(importances, s_method="TOPSIS", n_method="Linear1",
    #                      c_method="AbsPearson",
    #                      w_method="VIC", alt_names=alt_name)
    # else:
    #     rank = mcdm.rank(importances, s_method="TOPSIS", n_method="Linear1",
    #                      c_method="AbsPearson",
    #                      w_method="VIC", alt_names=alt_name)
    # selected = rank[:k]
    # choice_index = torch.LongTensor([int(i) for i, score in selected])
    # # info(f'current selection is {choice}')
    # choice = torch.zeros(fe.ds_size)
    # choice[choice_index] = 1
    # end = time.time()
    # info(f'the total infer time for MCDM is {end - start}')

#  gen 25
# 0.4341 [1. 1. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0.]
# 0.4357  [1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
#  0. 0.]
# 0.4301 gen 100

if __name__ == '__main__':
    main()
