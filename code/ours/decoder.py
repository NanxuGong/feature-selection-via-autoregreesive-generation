import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from feature_env import FeatureEvaluator
from utils.logger import info

SOS_ID = -1
EOS_ID = -1


class Attention(nn.Module):
    def __init__(self, input_dim, source_dim=None, output_dim=None, bias=False):
        super(Attention, self).__init__()
        if source_dim is None:
            source_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.source_dim = source_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, source_dim, bias=bias)
        self.output_proj = nn.Linear(input_dim + source_dim, output_dim, bias=bias)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, input, source_hids):
        batch_size = input.size(0)
        source_len = source_hids.size(1)

        # (batch, tgt_len, input_dim) -> (batch, tgt_len, source_dim)
        x = self.input_proj(input)

        # (batch, tgt_len, source_dim) * (batch, src_len, source_dim) -> (batch, tgt_len, src_len)
        attn = torch.bmm(x, source_hids.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, source_len), dim=1).view(batch_size, -1, source_len)

        # (batch, tgt_len, src_len) * (batch, src_len, source_dim) -> (batch, tgt_len, source_dim)
        mix = torch.bmm(attn, source_hids)

        # concat -> (batch, tgt_len, source_dim + input_dim)
        combined = torch.cat((mix, input), dim=2)
        # output -> (batch, tgt_len, output_dim)
        output = torch.tanh(self.output_proj(combined.view(-1, self.input_dim + self.source_dim))).view(batch_size, -1,
                                                                                                        self.output_dim)

        return output, attn


class Decoder(nn.Module):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length, gpu):
        super(Decoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.length = length  # total length to decode
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.sos_id = vocab_size - 1
        self.eos_id = vocab_size - 1
        self.gpu = gpu

    def forward(self, x, encoder_hidden=None, encoder_outputs=None):

        ret_dict = dict()
        ret_dict[Decoder.KEY_ATTN_SCORE] = list()
        if x is None:  # if not given x, then we are inferring!
            inference = True
        else:
            inference = False
        x, batch_size, length = self._validate_args(x, encoder_hidden,
                                                    encoder_outputs,
                                                    self.gpu)  # hidden is layer-wise out, out is final output
        assert length == self.length
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([length] * batch_size)

        def decode(step_, step_output_, step_attn_):
            decoder_outputs.append(step_output_)
            ret_dict[Decoder.KEY_ATTN_SCORE].append(step_attn_)
            # if step_ % 2 == 0:  # sample index, should be in [1, index-1]
            #     index = step_ // 2 % 10 // 2 + 3
            #     symbols_ = decoder_outputs[-1][:, 1:index].topk(1)[1] + 1
            # else:  # sample operation, should be in [7, 11]
            #     symbols_ = decoder_outputs[-1][:, 7:].topk(1)[1] + 7
            symbols_ = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols_)

            eos_batches = symbols_.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step_) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols_
        decoder_input = x[:, 0].unsqueeze(1)
        for di in range(length):
            if not inference:
                decoder_input = x[:, di].unsqueeze(1)
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                          encoder_outputs)
            step_output = decoder_output.squeeze(1)
            symbols = decode(di, step_output, step_attn)
            decoder_input = symbols

        ret_dict[Decoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[Decoder.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([h for h in encoder_hidden])
        else:
            encoder_hidden = encoder_hidden
        return encoder_hidden

    def _validate_args(self, x, encoder_hidden, encoder_outputs, gpu=0):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if x is None and encoder_hidden is None:
            batch_size = 1
        else:
            if x is not None:
                batch_size = x.size(0)
            else:
                batch_size = encoder_hidden[0].size(1)

        # set default input and max decoding length
        if x is None:
            x = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1).cuda(gpu)
            max_length = self.length
        else:
            max_length = x.size(1)

        return x, batch_size, max_length

    def infer(self, x, encoder_hidden=None, encoder_outputs=None):
        decoder_outputs, decoder_hidden, _ = self.forward(x, encoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden

    def forward_step(self, x: torch.Tensor, hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        pass


class RNNDecoder(Decoder):

    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length, gpu
                 ):
        super(RNNDecoder, self).__init__(
            layers,
            vocab_size,
            hidden_size,
            dropout,
            length, gpu)

        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.init_input = None
        self.attention = Attention(self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)  # class num + 1 eos

    def forward_step(self, x, hidden, encoder_outputs):
        batch_size = x.size(0)
        output_size = x.size(1)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        output, attn = self.attention(output, encoder_outputs)  # attention from decoder_output and encoder_output
        predicted_softmax = F.log_softmax(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1)
        predicted_softmax = predicted_softmax.view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

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

class TransformerDecoder(Decoder):
    def __init__(self, 
                 num_decoder_layers, 
                 nhead, 
                 vocab_size, 
                 embedding_size,
                 dropout, 
                 activation,
                 dim_feedforward,
                 batch_first,
                 length, 
                 gpu):
        super(TransformerDecoder, self).__init__(
            num_decoder_layers, 
            vocab_size, 
            embedding_size, 
            dropout, 
            length, 
            gpu)
        self.embedding_size = embedding_size
        # positional layer
        self.positionalEncoding = PositionalEncoding(
                                d_model = embedding_size,
                                dropout = dropout,
                                max_len = vocab_size)
        # decoder layer
        self.decoderLayer = nn.TransformerDecoderLayer(
                                d_model = embedding_size,
                                nhead = nhead,
                                dropout = dropout,
                                activation = activation,
                                dim_feedforward = dim_feedforward,
                                batch_first = batch_first)
        # stack decoder layer to construct transformer decoder
        self.decoder = nn.TransformerDecoder(
                                decoder_layer = self.decoderLayer,
                                num_layers = num_decoder_layers)
        self.attention = Attention(embedding_size)
        # out put
        self.out = nn.Linear(embedding_size, vocab_size)
        
    def forward_train_valid(self, x, encoder_outputs):
        batch_size = x.shape[0]
        output_size = x.shape[1]

        x = x.cuda(self.gpu)
        embedded = self.embedding(x)
        embedded = self.positionalEncoding(embedded)
        
        # construct squence mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz = output_size).cuda(self.gpu)
        out = self.decoder(embedded, encoder_outputs, tgt_mask)

        # out, attn = self.attention(out, encoder_outputs)

        predict_softmax = F.log_softmax(self.out(out.contiguous().view(-1, self.embedding_size)), dim=1)
        predict_softmax = predict_softmax.view(batch_size, output_size, -1)
        # print(predict_softmax)
        return predict_softmax

    def forward_step(self, encoder_outputs, input_id):
        embedded = self.embedding(input_id)
        embedded = self.positionalEncoding(embedded)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz = input_id.shape[1]).cuda(self.gpu)
        out = self.decoder(embedded, encoder_outputs, tgt_mask)

        # out, attn = self.attention(out, encoder_outputs)

        predict_softmax = F.log_softmax(self.out(out.contiguous().view(-1, self.embedding_size)), dim=1)
        _, next_input_id = predict_softmax.max(dim=1, keepdim=True)
        output_id = next_input_id.reshape(input_id.shape[0], input_id.shape[1])
        return output_id
    
    def forward_infer(self, encoder_outputs):
        # initialize the input id
        batch_size = encoder_outputs.shape[0]
        input_id = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1).cuda(self.gpu)
        for step in range(self.length):
            output_id = self.forward_step(encoder_outputs, input_id)
            input_id = torch.cat((input_id, output_id[:,-1].reshape(-1, 1)), dim=1)
        return output_id
        
def construct_decoder(fe: FeatureEvaluator, args) -> Decoder:
    name = args.method_name
    size = fe.ds_size
    info(f'Construct Decoder with method {name}...')
    if name == 'rnn':
        return RNNDecoder(
            layers=args.decoder_layers,
            vocab_size=size + 1,
            hidden_size=args.decoder_hidden_size,
            dropout=args.decoder_dropout,
            length=size,
            gpu=args.gpu
        )
    elif name == 'transformer' or name == 'transformerVae':
        return TransformerDecoder(
            num_decoder_layers = args.transformer_decoder_layers,
            nhead = args.decoder_nhead,
            vocab_size = size + 1,
            embedding_size = args.decoder_embedding_size,
            dropout = args.transformer_decoder_dropout,
            activation = args.transformer_decoder_activation,
            dim_feedforward = args.decoder_dim_feedforward,
            batch_first = args.batch_first,
            length = size,
            gpu = args.gpu
        )
    else:
        assert False
