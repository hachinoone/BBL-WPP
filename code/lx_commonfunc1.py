import pandas as pd
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import random
import torch
from torch import nn, optim
import os
import sys
import time
from torch.autograd import Variable
import math
from math import sqrt
import torch.nn.functional as F
from sklearn import metrics

device='cuda:3'

class Dataloader_self():
    def __init__(self, wfid, norm_method, window_size=60, pred_len=6, channel='physical'):
        # 导入原始数据并进行标准化
        # channel 可选择'physical', 'time', 和'wtnumber'
        self.wfid = wfid
        self.window_size = window_size
        self.pred_len = pred_len
        with open(r'wf' + str(wfid) + '/data_wf' + str(wfid) + '_10min', 'rb') as f:
            self.rawdata = pickle.load(f)
            self.rawdata = np.concatenate(self.rawdata, axis=0)
            self.rawdata = self.rawdata.reshape(self.rawdata.shape[0], -1, 13)
        if norm_method == 'maxmin':
            max_array = np.max(np.max(self.rawdata, axis=0), axis=0)
            min_array = np.min(np.min(self.rawdata, axis=0), axis=0)
            self.nume_term = min_array
            self.dom_term = max_array - min_array
        elif norm_method == 'meanstd':
            self.nume_term = np.mean(self.rawdata.reshape(-1, 13), axis=0)
            self.dom_term = np.std(self.rawdata.reshape(-1, 13), axis=0)
        self.norm_data = (self.rawdata - self.nume_term) / self.dom_term

        self.N_raw = self.norm_data.shape[0]
        self.wtnumber = self.norm_data.shape[1]
        self.channel = channel




    def data_split(self, target_use):
        # 将原始数据分割成对应部分 use = 'train', 'vali', 'test'
        N_train = int(self.N_raw * 0.6)
        N_vali = int(self.N_raw * 0.8)
        if target_use == 'train':
            return self.norm_data[:N_train, :, :]
        elif target_use == 'vali':
            return self.norm_data[N_train:N_vali, :, :]
        elif target_use == 'test':
            return self.norm_data[N_vali:, :, :]
    def dnn_data_generator(self, tgwtid, batch_size, target_use):
        # 生成数据，tgwtid是目标风机编号，batch_size是生成数据的batch大小，target_use是train/vali/test
        # 返回值为list_x和list_y，分别为list，list中每个元素为一个np格式的数组
        data = self.data_split(target_use)
        N_ = data.shape[0]

        x_rearrange_list = np.zeros([N_-self.window_size-self.pred_len, self.window_size, data.shape[1], data.shape[2]])
        y_rearrange_list = np.zeros([N_-self.window_size-self.pred_len, self.pred_len])
        for i in range(self.window_size):
            x_rearrange_list[:, i, :, :] = data[i:(-self.window_size-self.pred_len+i), :, :]
            #geo_rearrange_list[:, i, :, :] = geo[i:(-self.window_size-self.pred_len+i)]
        for i in range(self.pred_len):
            y_rearrange_list[:, i] = data[(i+self.window_size):(-self.pred_len+i), tgwtid, 0]
        # the shape so far: original: N_data, time, wtnumber, physical
        if self.channel == 'physical':
            x_rearrange_list = x_rearrange_list.swapaxes(1, 3)
        elif self.channel == 'wtnumber':
            x_rearrange_list = x_rearrange_list.swapaxes(1, 2)
        number_batch = int(np.ceil((N_-self.window_size-self.pred_len)/batch_size))
        list_x = list([])
        list_y = list([])
        for i in range(number_batch-1):
            list_x.append(x_rearrange_list[(batch_size*i):(batch_size*(i+1)), :, :, :])
            list_y.append(y_rearrange_list[(batch_size*i):(batch_size*(i+1)), :])
        list_x.append(x_rearrange_list[(batch_size*(number_batch-1)):, :, :, :])
        list_y.append(y_rearrange_list[(batch_size*(number_batch-1)):, :])

        return list_x, list_y
    def cnn_data_onesample(self):
        # 生成一个样本数据，主要用来返回data的形状
        data = self.norm_data[:self.window_size, :, :]

        # the shape so far: original: N_data, time, wtnumber, physical
        if self.channel == 'physical':
            data = data.swapaxes(0, 2)
        elif self.channel == 'wtnumber':
            data = data.swapaxes(0, 1)
        return data
    def dnn_data_onesample(self):
        # 生成一个样本数据，主要用来返回data的形状
        data = self.norm_data[:self.window_size].reshape(-1)

        return data


    def renorm(self, data):
        return data * self.dom_term[0] + self.nume_term[0]




class FullAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, mask=None):
        B, L, E = queries.shape
        _, S, D = values.shape
        scale = 1./sqrt(E)
        scores = torch.einsum("ble,bse->bls", queries, keys)
        if mask is not None:
            scores.masked_fill_(mask.to(values.device), -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bls,bsd->bld", A, values)
        
        return V.contiguous()

class ProbAttention(nn.Module):
    def __init__(self, time_len, n_wt, factor=2, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attention_dropout)
        mask_shape = [n_wt*n_wt, time_len, time_len]
        with torch.no_grad():
            self.mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)
            self.mask = self.mask.view(n_wt, n_wt, time_len, time_len).permute(2,0,3,1).contiguous().view(1, time_len*n_wt, time_len*n_wt)


    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, L, D]
        B, L_K, E = K.shape
        _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-2).expand(B, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None], M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k
        #print('jj', Q_K.size(), Q_reduce.size(), K.transpose(-2, -1).size(), M_top.size())

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, L_V, D = V.shape
        V_sum = V.mean(dim=1)
        contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, L, D = V.shape


        with torch.no_grad():

            mask_ex = self.mask.expand(B, L, L)
            indicator = mask_ex[torch.arange(B)[:, None], index, :].to(V.device)
            mask = indicator.view(scores.shape).to(V.device)
        scores.masked_fill_(mask, -np.inf)
        attn = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B)[:, None], index, :] = torch.matmul(attn, V)
        return context_in

    def forward(self, queries, keys, values):
        B, L_Q, D = queries.shape
        _, L_K, _ = keys.shape

        U_part = np.ceil(self.factor*np.sqrt(L_K)).astype('int').item() # c*sqrt(L_k)
        u = np.ceil(self.factor *np.sqrt(L_Q)).astype('int').item() # c*sqrt(L_q) 
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, u)
        # update the context with selected top_k queries


        context = self._update_context(context, values, scores_top, index, L_Q)
        
        return context.contiguous()

class AttnLayer(nn.Module):
    def __init__(self, time_len, n_wt, hidden_size, factor=1, attention_dropout=0.1):
        super(AttnLayer, self).__init__()
        self.attn1 = FullAttention(attention_dropout)
        self.attn2 = FullAttention(attention_dropout)
        self.attn3 = ProbAttention(time_len, n_wt, factor*4, attention_dropout)
        self.attn4 = ProbAttention(time_len, n_wt, factor, attention_dropout)
        self.project_out = nn.Sequential(nn.Linear(hidden_size*2, hidden_size*2), nn.Dropout(attention_dropout), nn.ReLU(), nn.Linear(hidden_size*2, hidden_size))
        mask_shape = [1, time_len, time_len]
        with torch.no_grad():
            self.mask1 = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)


    def forward(self, input):
        h1, h2 = [], []
        for i in range(input.size(1)):
            now = input[:, i, :, :]
            h = self.attn1(now, now, now)[:, None]
            h1.append(h)

        for i in range(input.size(2)):
            now = input[:, :, i, :]
            h = self.attn2(now, now, now, self.mask1)[:, :, None]
            h2.append(h)
        h1 = torch.cat(h1, dim=1)
        h2 = torch.cat(h2, dim=2)

        now = input.view(input.size(0), input.size(1)*input.size(2), input.size(3))
        #h3 = self.attn3(now, now, now).view(*input.size())
        #h4 = self.attn4(now, now, now).view(*input.size())
        #print(h1.size(), h2.size(), h3.size(), h4.size())

        return self.project_out(torch.cat((h1, h2), dim=-1))

class Attn(nn.Module):
    def __init__(self, time_len, n_wt, n_feat, hidden_size, pred_len, factor=0.5, attention_dropout=0.1, layer_num=3, device='cpu'):
        super(Attn, self).__init__()
        self.layer_num = layer_num
        self.device = device

        self.project_in = nn.Sequential(nn.Linear(n_feat, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size, bias=False))

        self.convs = []
        self.pools = []
        self.attns = []
        tl = time_len
        for i in range(layer_num):
            self.convs.append(nn.Conv1d(hidden_size, hidden_size, 4))
            self.pools.append(nn.MaxPool1d(2, 2))
            self.attns.append(AttnLayer(tl, n_wt, hidden_size, factor, attention_dropout))
            tl = int((tl-3-2)/2+1)
        self.convs = nn.ModuleList(self.convs)
        self.pools = nn.ModuleList(self.pools)
        self.attns = nn.ModuleList(self.attns)
        self.project_hidden = nn.Sequential(nn.Linear(hidden_size*tl, hidden_size*2), nn.Dropout(0.1), nn.ReLU(), nn.Linear(hidden_size*2, hidden_size))
        #self.lstm = nn.LSTM(input_size=hidden_size//2, num_layers=1, hidden_size=hidden_size//2, batch_first=True)
        self.project_out = nn.Linear(hidden_size, pred_len)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, wt, input, device):
        input = self.project_in(input)
        B, P = input.size(0), input.size(2)
        for i in range(self.layer_num):
            input = self.layer_norm1(self.attns[i](input))
            input = input.permute(0, 2, 3, 1).contiguous().reshape(B*P, input.size(3), input.size(1))
            input = self.pools[i](self.convs[i](input))
            input = self.layer_norm2(input.view(B, P, input.size(1), input.size(2)).permute(0, 3, 1, 2).contiguous())
        input = self.project_hidden(input[:, :, wt].reshape(input.size(0), -1))
        #x, (hn, cn) = self.lstm(input)
        return self.project_out(input)



class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input




class MHA(nn.Sequential):

    def __init__(self, n_heads, embed_dim, feed_forward_hidden=32, normalization='batch'):
        super(MHA, self).__init__()
        self.mha = MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim)
        self.norm1 = Normalization(embed_dim, normalization)
        self.project = nn.Sequential(nn.Linear(embed_dim, feed_forward_hidden), nn.ReLU(), nn.Linear(feed_forward_hidden, embed_dim))
        self.norm2 = Normalization(embed_dim, normalization)
    def forward(self, q):
        pq = q + self.norm1(self.mha(q))
        return self.norm2(pq + self.project(pq))


class Attn1(nn.Module):
    def __init__(self, time_len, n_wt, n_feat, hidden_size, pred_len, factor=0.5, attention_dropout=0.1, layer_num=2, n_heads=2, device='cpu'):
        super(Attn1, self).__init__()
        self.layer_num = layer_num
        self.device = device

        self.project_in = nn.Sequential(nn.Linear(n_feat, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size, bias=False))

        self.convs = []
        self.pools = []
        self.attns = []
        tl = time_len
        for i in range(layer_num):
            self.convs.append(nn.Conv1d(hidden_size, hidden_size, 4))
            self.pools.append(nn.MaxPool1d(2, 2))
            self.attns.append(MHA(n_heads, hidden_size*n_wt, hidden_size*n_wt))
            self.attns.append(MHA(n_heads, hidden_size*tl, hidden_size*tl))

            tl = int((tl-3-2)/2+1)
        self.convs = nn.ModuleList(self.convs)
        self.pools = nn.ModuleList(self.pools)
        self.attns = nn.ModuleList(self.attns)
        self.project_hidden = nn.Sequential(nn.Linear(hidden_size*tl, hidden_size*2), nn.Dropout(0.1), nn.ReLU(), nn.Linear(hidden_size*2, hidden_size))
        #self.lstm = nn.LSTM(input_size=hidden_size//2, num_layers=1, hidden_size=hidden_size//2, batch_first=True)
        self.project_out = nn.Linear(hidden_size, pred_len)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, wt, input, device):
        input = self.project_in(input)
        B, L, P, H = input.size(0), input.size(1), input.size(2), input.size(3)
        for i in range(self.layer_num):
            input = self.attns[2*i](input.view(B, L, -1)).view(B, L, P, H).permute(0, 2, 1, 3).contiguous()
            input = self.layer_norm1(self.attns[2*i+1](input.view(B, P, -1)).view(B, P, L, H))
            input = input.permute(0, 1, 3, 2).contiguous().reshape(B*P, H, L)
            input = self.pools[i](self.convs[i](input))

            L = int((L-3-2)/2+1)
            input = self.layer_norm2(input.view(B, P, H, L).permute(0, 3, 1, 2).contiguous())

        input = self.project_hidden(input[:, :, wt].reshape(input.size(0), -1))
        #x, (hn, cn) = self.lstm(input)
        return self.project_out(input)
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, pred_len, device='cpu'):
        super(DNN, self).__init__()
        self.project_out = nn.Sequential(nn.Linear(input_size, hidden_size*2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size*2, hidden_size), nn.ReLU(), nn.Dropout(0.1))
        self.final = nn.Linear(hidden_size, pred_len)
    def forward(self, input, device):
        self.emb = self.project_out(input.view(input.size(0), -1))
        return self.final(self.emb)
    def emb(self, input, device):
        return self.project_out(input.view(input.size(0), -1))


def data_save(data, save_path):
    data = pd.DataFrame(data)
    data.to_csv(save_path, index=None, header=None)
def RMSE_compute(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0))
def MAPE_compute(pred, true):
    return np.mean(np.abs(pred-true)/true, axis=0)
