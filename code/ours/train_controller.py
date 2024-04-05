import argparse
import os
import sys

import pandas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('/root/NIPSAutoFS/code/baseline')
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

from controller import GAFS
from feature_env import FeatureEvaluator, base_path
from utils_meter import AvgrageMeter, pairwise_accuracy, hamming_distance, count_parameters_in_MB, FSDataset
from record import SelectionRecord
from utils.logger import info, error
from utils.tools import test_task_new
import time
from transformers import AdamW, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser()
# Basic model parameters.

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--new_gen', type=int, default=200)
parser.add_argument('--method_name', type=str, choices=['rnn', 'transformer', 'transformerVae'], default='transformerVae')
parser.add_argument('--task_name', type=str, choices=['spectf', 'svmguide3', 'german_credit', 'spam_base',
                                                      'ionosphere', 'megawatt1', 'uci_credit_card', 'openml_618',
                                                      'openml_589', 'openml_616', 'openml_607', 'openml_620',
                                                      'openml_637', 'cifar-10', 'cifar-100', 'semeion',
                                                      'openml_586', 'uci_credit_card', 'higgs', 'ap_omentum_ovary','activity'
                                                      , 'mice_protein', 'coil-20', 'isolet', 'minist', 'minist_fashion'], default='svmguide3')
parser.add_argument('--gpu', type=int, default=0, help='used gpu')
parser.add_argument('--fe', type=str, choices=['+', '', '-'], default='-')
parser.add_argument('--top_k', type=int, default=25)
parser.add_argument('--gen_num', type=int, default=25)
parser.add_argument('--encoder_layers', type=int, default=1)
parser.add_argument('--encoder_hidden_size', type=int, default=64)
parser.add_argument('--encoder_emb_size', type=int, default=32)
parser.add_argument('--mlp_layers', type=int, default=2)
parser.add_argument('--mlp_hidden_size', type=int, default=200)
parser.add_argument('--decoder_layers', type=int, default=1)
parser.add_argument('--decoder_hidden_size', type=int, default=64)
# parser.add_argument('--source_length', type=int, default=40)
# parser.add_argument('--encoder_length', type=int, default=20)
# parser.add_argument('--decoder_length', type=int, default=40)
parser.add_argument('--encoder_dropout', type=float, default=0)
parser.add_argument('--mlp_dropout', type=float, default=0)
parser.add_argument('--decoder_dropout', type=float, default=0)
parser.add_argument('--l2_reg', type=float, default=0.0)
# parser.add_argument('--encoder_vocab_size', type=int, default=12)
# parser.add_argument('--decoder_vocab_size', type=int, default=12)
parser.add_argument('--max_step_size', type=int, default=100)
parser.add_argument('--trade_off', type=float, default=0.8)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--grad_bound', type=float, default=5.0)

parser.add_argument('--transformer_encoder_layers', type=int, default=2)
parser.add_argument('--encoder_nhead', type=int, default=8)
parser.add_argument('--encoder_embedding_size', type=int, default=64)
parser.add_argument('--transformer_encoder_dropout', type=float, default=0.1)
parser.add_argument('--transformer_encoder_activation', type=str, default='relu')
parser.add_argument('--encoder_dim_feedforward', type=int, default=128)
parser.add_argument('--batch_first', type=bool, default=True)  
parser.add_argument('--d_latent_dim', type=int, default=64)

parser.add_argument('--transformer_decoder_layers', type=int, default=2)
parser.add_argument('--decoder_nhead', type=int, default=8)
parser.add_argument('--transformer_decoder_dropout', type=float, default=0.1)
parser.add_argument('--transformer_decoder_activation', type=str, default='relu')
parser.add_argument('--decoder_dim_feedforward', type=int, default=128)
parser.add_argument('--decoder_embedding_size', type=int, default=64) 
parser.add_argument('--pre_train', type=str, default="True") 

parser.add_argument('--exp_name', type=str, default="")
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.3)
parser.add_argument('--gamma', type = float, default=0.2)
parser.add_argument('--eta', type=float, default=0.01)
parser.add_argument('--evaluate_redundancy', type = bool, default=True)
parser.add_argument('--pre_train_epochs', type=int, default=100)


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

def gafs_pre_train(train_queue, model: GAFS, optimizer, scheduler):
    # only train the decoder
    objs = AvgrageMeter()
    mse = AvgrageMeter()
    nll = AvgrageMeter()
    mse_r = AvgrageMeter()
    # kl = AvgrageMeter()

    model.train()
    for step, sample in enumerate(train_queue):
        encoder_input = sample['encoder_input']
        encoder_target = sample['encoder_target']
        encoder_redundancy = sample['encoder_redundancy']
        decoder_input = sample['decoder_input']
        decoder_target = sample['decoder_target']
        encoder_input = encoder_input.cuda(model.gpu)
        encoder_target = encoder_target.cuda(model.gpu).requires_grad_()
        encoder_redundancy = encoder_redundancy.cuda(model.gpu).requires_grad_()
        decoder_input = decoder_input.cuda(model.gpu)
        decoder_target = decoder_target.cuda(model.gpu)

        optimizer.zero_grad()
        predict_value, log_prob, arch, mu, logvar, predict_redundancy= model.forward(encoder_input, decoder_input)
        # 都没有除以batch size，如果需要除以batch size，令 reduction = "mean"
        loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze()) # mse loss
        loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1)) # ce loss
        if args.evaluate_redundancy:
            loss_3 = F.mse_loss(predict_redundancy.squeeze(), encoder_redundancy.squeeze())
            loss = args.alpha * loss_1 + args.beta * loss_2 + args.gamma * loss_3
        else:
            loss_3 = torch.tensor(0, dtype=torch.long)
            loss = (args.alpha + 0.2) * loss_1 + args.beta * loss_2 
        loss = loss.cuda(model.gpu)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        # print(lr)
        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)
        mse_r.update(loss_3, n)
    return objs.avg, mse.avg, nll.avg, mse_r.avg

def gafs_train(train_queue, model: GAFS, optimizer, scheduler):
    objs = AvgrageMeter()
    mse = AvgrageMeter()
    nll = AvgrageMeter()
    mse_r = AvgrageMeter()
    kl = AvgrageMeter()
    model.train()
    for step, sample in enumerate(train_queue):
        encoder_input = sample['encoder_input']
        encoder_target = sample['encoder_target']
        encoder_redundancy = sample['encoder_redundancy']
        decoder_input = sample['decoder_input']
        decoder_target = sample['decoder_target']
        encoder_input = encoder_input.cuda(model.gpu)
        encoder_target = encoder_target.cuda(model.gpu).requires_grad_()
        encoder_redundancy = encoder_redundancy.cuda(model.gpu).requires_grad_()
        decoder_input = decoder_input.cuda(model.gpu)
        decoder_target = decoder_target.cuda(model.gpu)

        optimizer.zero_grad()
        predict_value, log_prob, arch, mu, logvar, predict_redundancy= model.forward(encoder_input, decoder_input)
        # 都没有除以batch size，如果需要除以batch size，令 reduction = "mean"
        loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze()) # mse loss
        loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1)) # ce loss
        if args.evaluate_redundancy:
            loss_3 = F.mse_loss(predict_redundancy.squeeze(), encoder_redundancy.squeeze())
            if args.method_name == "transformerVae": 
                kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
                # if args.pre_train == 'True':
                #     loss = args.alpha * loss_1 + args.beta * loss_2 + args.gamma * loss_3
                # else:
                loss = args.alpha * loss_1 + args.beta * loss_2 + args.gamma * loss_3 + args.eta * kl_loss
            elif args.method_name == "transformer": 
                kl_loss = torch.tensor(1, dtype=torch.long)
                loss = args.alpha * loss_1 + args.beta * loss_2 + args.gamma * loss_3
                # loss = loss_1 + loss_2
                # l2_regularization = torch.tensor(0.0).cuda(model.gpu)
                # for param in model.parameters():
                #     l2_regularization += torch.norm(param, 2).cuda(model.gpu)
                # loss += args.l2_reg * l2_regularization 
            else:
                kl_loss = torch.tensor(1, dtype=torch.long)
                loss = args.alpha * loss_1 + args.beta * loss_2 + args.gamma * loss_3
        else:
            loss_3 = torch.tensor(0, dtype=torch.long)
            if args.method_name == "transformerVae": 
                kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
                # if args.pre_train == 'True':
                #     loss = (args.alpha + 0.2) * loss_1 + args.beta * loss_2 
                # else:
                loss = (args.alpha + 0.2) * loss_1 + args.beta * loss_2  + args.eta * kl_loss
            elif args.method_name == "transformer": 
                kl_loss = torch.tensor(1, dtype=torch.long)
                loss = (args.alpha + 0.2) * loss_1 + args.beta * loss_2 
                # loss = loss_1 + loss_2
                # l2_regularization = torch.tensor(0.0).cuda(model.gpu)
                # for param in model.parameters():
                #     l2_regularization += torch.norm(param, 2).cuda(model.gpu)
                # loss += args.l2_reg * l2_regularization 
            else:
                kl_loss = torch.tensor(1, dtype=torch.long)
                loss = (args.alpha + 0.2) * loss_1 + args.beta * loss_2 



        loss = loss.cuda(model.gpu)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()
        scheduler.step()
        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)
        mse_r.update(loss_3, n)
        kl.update(kl_loss.data, n)
    return objs.avg, mse.avg, nll.avg, mse_r.avg, kl.avg


def gafs_valid(queue, model: GAFS):
    pa = AvgrageMeter()
    pa_r = AvgrageMeter()
    hs = AvgrageMeter()
    mse = AvgrageMeter()
    mse_r = AvgrageMeter()
    ce = AvgrageMeter()
    with torch.no_grad():
        model.eval()
        for step, sample in enumerate(queue):
            encoder_input = sample['encoder_input']
            encoder_target = sample['encoder_target']
            encoder_redundancy = sample['encoder_redundancy']
            decoder_input = sample['decoder_input']
            decoder_target = sample['decoder_target']

            encoder_input = encoder_input.cuda(model.gpu)
            encoder_target = encoder_target.cuda(model.gpu)
            encoder_redundancy = encoder_redundancy.cuda(model.gpu)
            decoder_input = decoder_input.cuda(model.gpu)
            decoder_target = decoder_target.cuda(model.gpu)

            predict_value, logits, arch, mu, logvar,predict_redundancy = model.forward(encoder_input,decoder_input)
            n = encoder_input.size(0)
            pairwise_acc = pairwise_accuracy(encoder_target.data.squeeze().tolist(),
                                             predict_value.data.squeeze().tolist())
            if args.evaluate_redundancy:
                pairwise_acc_r = pairwise_accuracy(encoder_redundancy.squeeze().tolist(),
                                                predict_redundancy.squeeze().tolist())
                mse_r.update(F.mse_loss(predict_redundancy.data.squeeze(), encoder_redundancy.data.squeeze()))
            else:
                pairwise_acc_r = 0
                mse_r.update(torch.tensor(0, dtype=torch.long))
            hamming_dis = hamming_distance(decoder_target.data.squeeze().tolist(), arch.data.squeeze().tolist())
            mse.update(F.mse_loss(predict_value.data.squeeze(), encoder_target.data.squeeze()), n)
            
            pa.update(pairwise_acc, n)
            pa_r.update(pairwise_acc_r,n)
            hs.update(hamming_dis, n)
            ce.update(F.nll_loss(logits.contiguous().view(-1, logits.size(-1)), decoder_target.view(-1)), n)
    return mse.avg, mse_r.avg, pa.avg, pa_r.avg, hs.avg, ce.avg


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


def gafs_infer(queue, model, step, direction='+', evaluate_redundancy = False):
    new_gen_list = []
    model.eval()
    for i, sample in enumerate(queue):
        encoder_input = sample['encoder_input']
        encoder_redundancy = sample['encoder_redundancy']
        encoder_input = encoder_input.cuda(model.gpu)
        # print(encoder_input.shape)
        model.zero_grad()
        new_gen = model.generate_new_feature(encoder_input, predict_lambda=step, direction=direction, evaluate_redundancy = evaluate_redundancy)
        new_gen_list.extend(new_gen.data.squeeze().tolist())
    return new_gen_list


def select_top_k_r(choice: Tensor, labels: Tensor, redundancy: Tensor, k: int) -> (Tensor, Tensor, Tensor):
    redundancy = torch.tensor(redundancy).unsqueeze(-1)
    performance = labels - 0.1 * redundancy
    values, indices = torch.topk(performance, k, dim=0)
    # print()
    return choice[indices.squeeze()], labels[indices.squeeze()], redundancy[indices.squeeze()]

def select_top_k(choice: Tensor, labels: Tensor, redundancy: Tensor, k: int) -> (Tensor, Tensor, Tensor):
    redundancy = torch.tensor(redundancy).unsqueeze(-1)
    values, indices = torch.topk(labels, k, dim=0)
    return choice[indices.squeeze()], labels[indices.squeeze()], redundancy[indices.squeeze()]

def get_redundancy(choice, fe):
    rm = fe.redundancy_matrix
    redundancy = 0
    for i in range(0, choice.shape[0]):
        if choice[i] == 0:
            continue
        j = i + 1
        while j < choice.shape[0]:
            if choice[j] == 0:
                j += 1
                continue
            redundancy = redundancy + rm[i][j]
            j += 1
            
    return redundancy

def main():
    if not torch.cuda.is_available():
        info('No GPU found!')
        sys.exit(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
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
    if args.pre_train == "False":
        model.load_state_dict(torch.load(f'{base_path}/history/{args.task_name}/GAFS_pretrain_{args.method_name}_{str(args.exp_name)}.model_dict'))
    elif args.pre_train == "Search":
        model.load_state_dict(torch.load(f'{base_path}/history/{args.task_name}/GAFS_{args.method_name}_{str(args.exp_name)}.model_dict'))
        model.load_state_dict(torch.load(f'{base_path}/history/{args.task_name}/GAFS_{args.method_name}_.model_dict'))
    
    info(f"param size = {count_parameters_in_MB(model)}MB")
    model = model.cuda(device)
    # 设置为0表示不进行数据增强
    choice, redundancies, labels = fe.get_record(args.gen_num, eos=fe.ds_size)
    valid_choice, valid_redundancies, valid_labels = fe.get_record(0, eos=fe.ds_size)   
    min_val = min(labels)
    max_val = max(labels)
    train_encoder_target = [(i - min_val) / (max_val - min_val) for i in labels]
    valid_encoder_target = [(i - min_val) / (max_val - min_val) for i in valid_labels]
    # min_r_val = min(redundancies)
    max_r_val = get_redundancy(np.ones(choice.shape[1]), fe)
    train_encoder_redundancy = [i / max_r_val for i in redundancies]
    valid_encoder_redundancy = [i / max_r_val for i in valid_redundancies]
    train_dataset = FSDataset(choice, train_encoder_redundancy, train_encoder_target, train=True, sos_id=fe.ds_size, eos_id=fe.ds_size)
    valid_dataset = FSDataset(valid_choice,  valid_encoder_redundancy, valid_encoder_target, train=True, sos_id=fe.ds_size, eos_id=fe.ds_size)
    train_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_queue = torch.utils.data.DataLoader(
        valid_dataset, batch_size=len(valid_dataset), shuffle=False, pin_memory=True)

    if args.pre_train == 'True' and args.method_name == 'transformerVae':
        info('Begin to pre-train')
       
        nao_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
        len_dataset = len(train_dataset)
        total_steps = (len_dataset // args.batch_size) * args.pre_train_epochs if len_dataset % args.batch_size == 0 else (len_dataset // args.batch_size + 1) * args.pre_train_epochs # 每一个epoch中有多少个step可以根据len(DataLoader)计算：total_steps = len(DataLoader) * epoch
        warm_up_ratio = 0.1 # 定义要预热的step
        scheduler = get_linear_schedule_with_warmup(nao_optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)

        save_model = model
        cur_loss = float('inf')
        best_epoch = 0
        for nao_epoch in range(1, args.pre_train_epochs + 1):
            sys.stdout.flush()
            sys.stderr.flush()
            nao_loss, nao_mse, nao_ce, nao_mse_r = gafs_pre_train(train_queue, model, nao_optimizer,scheduler)
            info("epoch {:04d}, train loss {:.6f}, mse {:.6f}, ce {:.6f}, mse_r {:.6f}".format(nao_epoch, nao_loss, nao_mse, nao_ce, nao_mse_r))
            if nao_loss < cur_loss:
                save_model = model
                cur_loss = nao_loss
                best_epoch = nao_epoch
            # if cur_loss < args.early_stop:
            #     save_model = model
            #     cur_loss = nao_loss
            #     best_epoch = nao_epoch
            #     break
        model = save_model
        info("best model from epoch {:04d}, loss is {:.6f}".format(best_epoch,cur_loss))

    if args.pre_train !='Search':
        info('Training Encoder-Predictor-Decoder')
        # nao_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
        nao_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
        len_dataset = len(train_dataset)
        total_steps = (len_dataset // args.batch_size) * args.epochs if len_dataset % args.batch_size == 0 else (len_dataset // args.batch_size + 1) * args.epochs # 每一个epoch中有多少个step可以根据len(DataLoader)计算：total_steps = len(DataLoader) * epoch
        warm_up_ratio = 0.1 # 定义要预热的step
        if args.method_name == "transformer":
            scheduler = get_linear_schedule_with_warmup(nao_optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)
        else:
            scheduler = get_linear_schedule_with_warmup(nao_optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
        save_model = model
        cur_loss = float('inf')
        best_epoch = 0
        for nao_epoch in range(1, args.epochs + 1):
            sys.stdout.flush()
            sys.stderr.flush()
            nao_loss, nao_mse, nao_ce, nao_mse_r, kl = gafs_train(train_queue, model, nao_optimizer, scheduler)
            if nao_epoch % 10 == 0 or nao_epoch == 1:
                info("epoch {:04d} train loss {:.6f} mse {:.6f} ce {:.6f} mse_r {:.6f} kl {:.6f}".format(nao_epoch, nao_loss, nao_mse, nao_ce, nao_mse_r, kl))
                if nao_loss < cur_loss:
                    save_model = model
                    cur_loss = nao_loss
                    best_epoch = nao_epoch
            if nao_epoch % 100 == 0 or nao_epoch == 1:
                mse, mse_r, pa, pa_r, hs, ce = gafs_valid(train_queue, model)
                info("Evaluation on train data")
                info('epoch {:04d} mse {:.6f} mse_r {:.6f} ce {:.6f} pairwise accuracy {:.6f} pairwise accuracy_r {:.6f} hamming distance {:.6f}'
                    .format(nao_epoch, mse, mse_r, ce, pa, pa_r, hs))
                mse, mse_r, pa, pa_r, hs, ce = gafs_valid(valid_queue, model)
                info("Evaluation on valid data")
                info('epoch {:04d} mse {:.6f} mse_r {:.6f} ce {:.6f} pairwise accuracy {:.6f} pairwise accuracy_r {:.6f} hamming distance {:.6f}'
                    .format(nao_epoch, mse, mse_r, ce, pa, pa_r, hs))
        model = save_model
        info("best model from epoch {:04d}".format(best_epoch))

    info("Begin to Search")
    if args.evaluate_redundancy:
        top_selection, top_performance, top_redundancy = select_top_k_r(valid_choice, valid_labels, valid_encoder_redundancy, args.top_k)
    else:
        top_selection, top_performance, top_redundancy = select_top_k(valid_choice, valid_labels, valid_encoder_redundancy, args.top_k)
    # print(top_selection.shape)
    infer_dataset = FSDataset(top_selection, top_redundancy, top_performance, False, sos_id=fe.ds_size, eos_id=fe.ds_size)
    infer_queue = DataLoader(infer_dataset, batch_size=len(infer_dataset), shuffle=False,
                             pin_memory=True)
    # if args.method_name != "transformerVae" or (args.method_name == "transformerVae" and args.pre_train != "True"):
    new_selection = []
    new_choice = []
    new_redundancy = []
    predict_step_size = 0
    while len(new_selection) < args.new_gen:
        predict_step_size += 1
        info('Generate new architectures with step size {:.2f}'.format(predict_step_size))
        # timestamp = int(time.time()*1000)
        new_record = gafs_infer(infer_queue, model, direction='+', step=predict_step_size, evaluate_redundancy = args.evaluate_redundancy)
        # new_timestamp = int(time.time()*1000)
        # time_cost = new_timestamp - timestamp
        # print("time cost:")
        # print(time_cost)
        # break
        for choice in new_record:
            onehot_choice = choice_to_onehot(choice)
            if onehot_choice.sum() <= 0:
                error('insufficient selection')
                continue
            record = SelectionRecord(onehot_choice.numpy(), -1)
            if record not in fe.records.r_list and record not in new_selection:
                new_selection.append(record)
                new_choice.append(onehot_choice)
                redundancy = get_redundancy(onehot_choice, fe) / max_r_val
                new_redundancy.append(redundancy)
            if len(new_selection) >= args.new_gen:
                break
        info(f'{len(new_selection)} new choice generated now', )
        if predict_step_size > args.max_step_size:
            break
    info(f'build {len(new_selection)} new choice !!!')

    new_choice_pt = torch.stack(new_choice)
    choice_path = f'{base_path}/history/{fe.task_name}/generated_choice_{args.method_name}_{str(args.exp_name)}.pt'
    torch.save(new_choice_pt, choice_path)
    info(f'save generated choice to {choice_path}')

    previous_optimal = float(torch.max(valid_labels))
    optimal_selection = None
    if args.pre_train == "True":
        torch.save(model.state_dict(), f'{base_path}/history/{fe.task_name}/GAFS_pretrain_{args.method_name}_{str(args.exp_name)}.model_dict')
        torch.save(model.state_dict(), f'{base_path}/history/{fe.task_name}/GAFS_{args.method_name}_{str(args.exp_name)}.model_dict')
    else:
        torch.save(model.state_dict(), f'{base_path}/history/{fe.task_name}/GAFS_{args.method_name}_{str(args.exp_name)}.model_dict')
    # return -1

    best_selection = None
    best_redundancy = None
    best_optimal = -1000
    best_selection_test = None
    best_redundancy_test = None
    best_optimal_test = -1000
    # info(f'the best performance for this task is {previous_optimal}')
    for s, r in zip(new_selection, new_redundancy):
        train_data = fe.generate_data(s.operation, 'train')
        result = fe.get_performance(train_data)
        if args.evaluate_redundancy:
            performance = result - 0.1 * r
        else:
            performance = result
        test_data = fe.generate_data(s.operation, 'test')
        test_result = fe.get_performance(test_data)
        if args.evaluate_redundancy:
            test_performance = test_result - 0.1 * r
        else:
            test_performance = test_result
        # if result > previous_optimal:
        #     optimal_selection = s.operation
        #     previous_optimal = result
        #     info(f'found optimal selection! the choice is {s.operation}, the performance on train is {result}')
        if performance > best_optimal:
            best_selection = s.operation
            best_redundancy = r.item()
            best_optimal = performance
            info(f'found best on train : {best_optimal}')
        if test_performance > best_optimal_test:
            best_selection_test = s.operation
            best_redundancy_test = r.item()
            best_optimal_test = test_performance
            info(f'found best on test : {best_optimal_test}')

    opt_path = f'{base_path}/history/{fe.task_name}/best-ours.hdf'
    best_num = np.sum(best_selection)
    best_ori = fe.generate_data(best_selection, 'test')
    ori_p = fe.get_performance(best_ori)
    # _,_,ori_p,_=test_task_new(best_ori,task = fe.task_type)
    # ori_p = fe.report_performance(best_selection, flag='test')
    info(f'found train generation in our method! the choice is {best_selection}, the performance is {ori_p}')
    fe.generate_data(best_selection, 'train').to_hdf(opt_path, key='train')
    fe.generate_data(best_selection, 'test').to_hdf(opt_path, key='test')

    opt_path_test = f'{base_path}/history/{fe.task_name}/best-ours-test.hdf'
    best_test_num = np.sum(best_selection_test)
    # test_p = fe.report_performance(best_selection_test, flag='test')
    best_test = fe.generate_data(best_selection_test, flag='test')
    test_p = fe.get_performance(best_test)
    # _,_,test_p,_=test_task_new(best_test,task = fe.task_type)
    info(f'found test generation in our method! the choice is {best_selection_test}, the performance is {test_p}')
    fe.generate_data(best_selection_test, 'train').to_hdf(opt_path_test, key='train')
    fe.generate_data(best_selection_test, 'test').to_hdf(opt_path_test, key='test')
    ps = []
    info('given overall validation')
    report_head = 'RAW\t'
    report_num = ''
    report_r = ''
    raw_test = pandas.read_hdf(f'{base_path}/history/{fe.task_name}.hdf', key='raw_test')
    ps.append('{:.2f}'.format(fe.get_performance(raw_test) * 100))
    # number of feature
    raw_num = raw_test.shape[1]-1
    report_num += '{:d}\t'.format(raw_num)
    report_r += '{:.4f}\t'.format(1)
    for method in baseline_name:
        report_head += f'{method}\t'
        spe_test = pandas.read_hdf(f'{base_path}/history/{fe.task_name}.hdf', key=f'{method}_test')
        spe_choice = pandas.read_hdf(f'{base_path}/history/{fe.task_name}.hdf', key=f'{method}_choice')
        spe_redundancy = get_redundancy(spe_choice.values, fe)
        spe_num = spe_test.shape[1]-1
        report_num += '{:d}\t'.format(spe_num)
        report_r += '{:.4f}\t'.format((spe_redundancy / max_r_val).item())
        ps.append('{:.2f}'.format(fe.get_performance(spe_test) * 100))
    report_head += 'Ours\tOurs_Test'
    report = ''
    print(report_head)
    for per in ps:
        report += f'{per}%\t'
    report_num += '{:d}\t'.format(int(best_num))
    report_num += '{:d}\t'.format(int(best_test_num))
    report_r += '{:.4f}\t'.format(best_redundancy)
    report_r += '{:.4f}\t'.format(best_redundancy_test)
    report += '{:.2f}%\t'.format(ori_p * 100)
    report += '{:.2f}%\t'.format(test_p * 100)
    print(report)
    print(report_r)
    print(report_num)


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