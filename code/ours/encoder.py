from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('/root/NIPSAutoFS/code/baseline')
from feature_env import FeatureEvaluator
from utils.logger import info


class Encoder(nn.Module):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size):
        super().__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)

    def infer(self, x, predict_lambda, direction='-', evaluate_redundancy = False):
        encoder_outputs, encoder_hidden, seq_emb, predict_value, mu, logvar, predict_redundancy = self(x)
        grads_value = torch.autograd.grad(predict_value, encoder_outputs, torch.ones_like(predict_value),retain_graph=True)[0]
        grads_redundancy = 0
        if evaluate_redundancy:
            grads_redundancy = torch.autograd.grad(predict_redundancy, encoder_outputs, torch.ones_like(predict_value))[0]
        if direction == '+':
            new_encoder_outputs = encoder_outputs + predict_lambda * grads_value - predict_lambda * 0.1 * grads_redundancy
        elif direction == '-':
            new_encoder_outputs = encoder_outputs - predict_lambda * grads_value + predict_lambda * 0.1 * grads_redundancy
        else:
            raise ValueError('Direction must be + or -, got {} instead'.format(direction))
        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        new_seq_emb = torch.mean(new_encoder_outputs, dim=1)
        new_seq_emb = F.normalize(new_seq_emb, 2, dim=-1)
        return encoder_outputs, encoder_hidden, seq_emb, predict_value, new_encoder_outputs, new_seq_emb

    def forward(self, x):
        pass


class RNNEncoder(Encoder):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 mlp_layers,
                 mlp_hidden_size,
                 mlp_dropout
                 ):
        super(RNNEncoder, self).__init__(layers, vocab_size, hidden_size)

        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = mlp_hidden_size

        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential()
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
            else:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
        
        self.mlp_r = nn.Sequential()
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp_r.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
            else:
                self.mlp_r.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
        self.regressor = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)
        self.regressor_r = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)

    def forward(self, x):
        embedded = self.embedding(x)  # batch x length x hidden_size
        embedded = self.dropout(embedded)
        # print(x.shape)
        # hidden是一个tuple(h_n, c_n)
        out, hidden = self.rnn(embedded)
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs = out  # final output
        encoder_hidden = hidden  # layer-wise hidden

        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        seq_emb = out
        out_ = out
        out = self.mlp(out)
        out = self.regressor(out)
        predict_value = torch.sigmoid(out)
        mu = 0
        logvar = 0 
        #redundancy
        out_ = self.mlp_r(out_)
        out_ = self.regressor_r(out_)
        predict_redundancy = torch.sigmoid(out_)
        return encoder_outputs, encoder_hidden, seq_emb, predict_value, mu, logvar, predict_redundancy

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs,例如(1,7, 128),batch size为1,7个单词,单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
class TransformerEncoder(Encoder):
    def __init__(
            self,
            num_encoder_layers,
            nhead, 
            vocab_size,
            embedding_size,
            dropout, 
            activation,
            dim_feedforward,
            batch_first,
            mlp_layers,
            mlp_hidden_size,
            mlp_dropout
            ):
        super(TransformerEncoder, self).__init__(num_encoder_layers, vocab_size, embedding_size)
        # positional layer
        self.positionalEncoding = PositionalEncoding(
                                d_model = embedding_size,
                                dropout = dropout,
                                max_len = vocab_size)
        # multi-head attention && feed forward && norm -> encoder layer 
        self.encoderLayer = nn.TransformerEncoderLayer(
                                d_model = embedding_size,
                                nhead = nhead,
                                dropout = dropout,
                                activation = activation,
                                dim_feedforward = dim_feedforward,
                                batch_first = batch_first)
        # stack encoder layers to construct transformer encoder
        self.encoder = nn.TransformerEncoder(
                                encoder_layer = self.encoderLayer,
                                num_layers = num_encoder_layers)
        # mlp layer
        self.mlp = nn.Sequential()
        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = mlp_hidden_size
        for i in range(mlp_layers):
            if i == 0:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(embedding_size, mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
            else:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(mlp_hidden_size, mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
        self.mlp_r = nn.Sequential()
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp_r.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
            else:
                self.mlp_r.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
        self.regressor = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)
        self.regressor_r = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)

    def forward(self, x):
        # get embedding
        embedding = self.embedding(x)
        # add positional information
        embedding = self.positionalEncoding(embedding)
        
        # encoder output
        out = self.encoder(embedding)
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs = out
        
        # add all embedding and compute mean
        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        seq_emb = out
        
        out_ = out
        # evaluator
        out = self.mlp(out)
        out = self.regressor(out)
        predict_value = torch.sigmoid(out)
        # 适配RNNEncoder的输出
        encoder_hidden = None
        mu = 0
        logvar = 0

        out_ = self.mlp_r(out_)
        out_ = self.regressor_r(out_)
        predict_redundancy = torch.sigmoid(out_)

        # encoder_outputs, encoder_hidden, seq_emb, predict_value
        # encoder_outputs shape (batch_size, sequence_length, embedding_dim)
        return encoder_outputs, encoder_hidden, seq_emb, predict_value, mu, logvar,predict_redundancy

class TransformerEncoderVAE(Encoder):
    def __init__(
            self,
            num_encoder_layers,
            nhead, 
            vocab_size,
            embedding_size,
            dropout, 
            activation,
            dim_feedforward,
            batch_first,
            mlp_layers,
            mlp_hidden_size,
            mlp_dropout,
            d_latent_dim,
            ):
        super(TransformerEncoderVAE, self).__init__(num_encoder_layers, vocab_size, embedding_size)
        # positional layer
        self.positionalEncoding = PositionalEncoding(
                                d_model = embedding_size,
                                dropout = dropout,
                                max_len = vocab_size)
        # multi-head attention && feed forward && norm -> encoder layer 
        self.encoderLayer = nn.TransformerEncoderLayer(
                                d_model = embedding_size,
                                nhead = nhead,
                                dropout = dropout,
                                activation = activation,
                                dim_feedforward = dim_feedforward,
                                batch_first = batch_first)
        # stack encoder layers to construct transformer encoder
        self.encoder = nn.TransformerEncoder(
                                encoder_layer = self.encoderLayer,
                                num_layers = num_encoder_layers)
        self.mu = nn.Linear(embedding_size, d_latent_dim)
        self.logvar = nn.Linear(embedding_size, d_latent_dim)
        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = mlp_hidden_size
        # mlp layer
        self.mlp = nn.Sequential()
        for i in range(mlp_layers):
            if i == 0:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(d_latent_dim, mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
            else:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(mlp_hidden_size, mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
        self.mlp_r = nn.Sequential()
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp_r.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
            else:
                self.mlp_r.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
        self.regressor = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)
        self.regressor_r = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)
    
    def reparameterize(self, mu, logvar):
        # epsilon: 噪声
        # epsilon = torch.randn_like(mu)
        epsilon = 1
        return mu + epsilon * torch.exp(logvar/2)
    
    def forward(self, x):
        # get embedding
        embedding = self.embedding(x)
        # add positional information
        embedding = self.positionalEncoding(embedding)
        
        # encoder output
        out = self.encoder(embedding)
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs = out
        
        # add all embedding and compute mean / summarize
        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        seq_emb = out
        
        # compute mu, logvar to compute the KL-loss
        mu, logvar = self.mu(out), self.logvar(out)
        # reparameterize
        out = self.reparameterize(mu, logvar)
        
        out_ = out
        # evaluator
        out = self.mlp(out)
        out = self.regressor(out)
        predict_value = torch.sigmoid(out)

        out_ = self.mlp_r(out_)
        out_ = self.regressor_r(out_)
        predict_redundancy = torch.sigmoid(out_)

        # 适配RNNEncoder的输出
        encoder_hidden = None
        # encoder_outputs, encoder_hidden, seq_emb, predict_value
        # encoder_outputs shape (batch_size, sequence_length, embedding_dim)
        return encoder_outputs, encoder_hidden, seq_emb, predict_value, mu, logvar,predict_redundancy


def construct_encoder(fe: FeatureEvaluator, args) -> Encoder:
    name = args.method_name
    size = fe.ds_size
    info(f'Construct Encoder with method {name}...')
    if name == 'rnn':
        return RNNEncoder(
            layers=args.encoder_layers,
            vocab_size=size + 1,
            hidden_size=args.encoder_hidden_size,
            dropout=args.encoder_dropout,
            mlp_layers=args.mlp_layers,
            mlp_hidden_size=args.mlp_hidden_size,
            mlp_dropout=args.encoder_dropout
        )
    elif name == 'transformer':
        return TransformerEncoder(
            num_encoder_layers = args.transformer_encoder_layers, # default: 6
            nhead = args.encoder_nhead, # default: 8
            vocab_size = size + 1, # num_size + <begin> + <end> + <padding>
            embedding_size = args.encoder_embedding_size, # default: 64
            dropout = args.transformer_encoder_dropout, # default: 0.1
            activation = args.transformer_encoder_activation, # default: relu
            dim_feedforward = args.encoder_dim_feedforward, # default: 2048
            batch_first = args.batch_first,  # default: True
            mlp_layers = args.mlp_layers, # default: 2
            mlp_hidden_size = args.mlp_hidden_size, # default: 200
            mlp_dropout = args.encoder_dropout # default: 0
        )
    elif name == 'transformerVae':
        return TransformerEncoderVAE(
            num_encoder_layers = args.transformer_encoder_layers,
            nhead = args.encoder_nhead,
            vocab_size = size + 1,
            embedding_size = args.encoder_embedding_size,
            dropout = args.transformer_encoder_dropout,
            activation = args.transformer_encoder_activation,
            dim_feedforward = args.encoder_dim_feedforward,
            batch_first = args.batch_first,
            mlp_layers = args.mlp_layers,
            mlp_hidden_size = args.mlp_hidden_size,
            mlp_dropout = args.encoder_dropout,
            d_latent_dim = args.d_latent_dim
        )
    else:
        assert False
