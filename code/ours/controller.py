import torch
import torch
import torch.nn as nn
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('/root/NIPSAutoFS/code/baseline')
from decoder import construct_decoder
from encoder import construct_encoder
from feature_env import FeatureEvaluator

SOS_ID = 0
EOS_ID = 0


# gradient based automatic feature selection
class GAFS(nn.Module):
    def __init__(self,
                 fe:FeatureEvaluator,
                 args
                 ):
        super(GAFS, self).__init__()
        self.style = args.method_name
        self.gpu = args.gpu
        self.encoder = construct_encoder(fe, args)
        self.decoder = construct_decoder(fe, args)
        if self.style == 'rnn':
            self.flatten_parameters()
            
    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, target_variable=None):
        mu = 0.0
        logvar = 0.0
        if self.style == 'rnn':
            encoder_outputs, encoder_hidden, feat_emb, predict_value, mu, logvar, redundancy = self.encoder.forward(input_variable)
            decoder_hidden = (feat_emb.unsqueeze(0), feat_emb.unsqueeze(0))
            decoder_outputs, decoder_hidden, ret = self.decoder.forward(target_variable, decoder_hidden, encoder_outputs)
            decoder_outputs = torch.stack(decoder_outputs, 0).permute(1, 0, 2)
            feat = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        # transformer可以并行，不需要step by step
        elif self.style == 'transformer':
            # 适配vae
            encoder_outputs, encoder_hidden, feat_emb, predict_value, mu, logvar, redundancy = self.encoder.forward(input_variable)
            decoder_outputs = self.decoder.forward_train_valid(target_variable, encoder_outputs)
            _, feat = decoder_outputs.max(2, keepdim=True)
            feat = feat.reshape(input_variable.size(0), input_variable.size(1))
        elif self.style == "transformerVae":
            encoder_outputs, encoder_hidden, feat_emb, predict_value, mu, logvar, redundancy = self.encoder.forward(input_variable)
            decoder_outputs = self.decoder.forward_train_valid(target_variable, encoder_outputs)
            _, feat = decoder_outputs.max(2, keepdim=True)
            feat = feat.reshape(input_variable.size(0), input_variable.size(1))

        return predict_value, decoder_outputs, feat, mu, logvar, redundancy


    def generate_new_feature(self, input_variable, predict_lambda=1, direction='-', evaluate_redundancy = False):
        if self.style == 'rnn':
            encoder_outputs, encoder_hidden, feat_emb, predict_value, new_encoder_outputs, new_feat_emb = \
                self.encoder.infer(input_variable, predict_lambda, direction=direction, evaluate_redundancy = evaluate_redundancy)
            new_encoder_hidden = (new_feat_emb.unsqueeze(0), new_feat_emb.unsqueeze(0))
            decoder_outputs, decoder_hidden, ret = self.decoder.forward(None, new_encoder_hidden, new_encoder_outputs)
            new_feat_seq = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        elif self.style == 'transformer' or self.style == 'transformerVae':
            # 根据梯度搜索出的新的encode输出
            encoder_outputs, encoder_hidden, feat_emb, predict_value, new_encoder_outputs, new_feat_emb = \
                self.encoder.infer(input_variable, predict_lambda, direction=direction, evaluate_redundancy = evaluate_redundancy)
            new_feat_seq = self.decoder.forward_infer(new_encoder_outputs)
        return new_feat_seq
