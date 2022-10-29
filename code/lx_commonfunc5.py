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
import torch.nn.functional as F
from sklearn import metrics
from math import sqrt

device='cuda:1'

class Dataloader_self():
    def __init__(self, wfid, norm_method, window_size=60, pred_len=6, channel='physical', scale=85):
        # 导入原始数据并进行标准化
        # channel 可选择'physical', 'time', 和'wtnumber'
        self.wfid = wfid
        self.window_size = window_size
        self.pred_len = pred_len
        with open(r'wf' + str(wfid) + '/data_wf' + str(wfid) + '_7s', 'rb') as f:
            self.rawdata = pickle.load(f)
        with open(r'wf' + str(wfid) + '/data_wf' + str(wfid) + '_10min', 'rb') as f:
            self.rawdata_10 = pickle.load(f)
            self.rawdata_10 = np.concatenate(self.rawdata_10, axis=0)
            self.rawdata_10 = self.rawdata_10.reshape(self.rawdata_10.shape[0], -1, 13)
        if norm_method == 'maxmin':
            max_array = np.max(np.max(self.rawdata_10, axis=0), axis=0)
            min_array = np.min(np.min(self.rawdata_10, axis=0), axis=0)
            self.nume_term = min_array
            self.dom_term = max_array - min_array
            max_array = np.max(np.max(self.rawdata, axis=0), axis=0)
            min_array = np.min(np.min(self.rawdata, axis=0), axis=0)
            nume_term = min_array
            dom_term = max_array - min_array

        elif norm_method == 'meanstd':
            self.nume_term = np.mean(self.rawdata.reshape(-1, 13), axis=0)
            self.dom_term = np.std(self.rawdata.reshape(-1, 13), axis=0)
        self.norm_data_10 = (self.rawdata_10 - self.nume_term) / self.dom_term
        self.norm_data = (self.rawdata - nume_term) / dom_term
        self.N_raw = self.norm_data_10.shape[0]
        self.wtnumber = self.norm_data.shape[1]
        self.channel = channel
        self.scale = scale


    def data_split(self, target_use):
        # 将原始数据分割成对应部分 use = 'train', 'vali', 'test'
        N_train = int(self.N_raw * 0.6)
        N_vali = int(self.N_raw * 0.8)
        if target_use == 'train':
            return self.norm_data[:N_train*self.scale, :, :], self.norm_data_10[:N_train, :, :]
        elif target_use == 'vali':
            return self.norm_data[N_train*self.scale:N_vali*self.scale, :, :], self.norm_data_10[N_train:N_vali, :, :]
        elif target_use == 'test':
            return self.norm_data[N_vali*self.scale:, :, :], self.norm_data_10[N_vali:, :, :]
    def dnn_data_generator(self, tgwtid, batch_size, target_use):
        # 生成数据，tgwtid是目标风机编号，batch_size是生成数据的batch大小，target_use是train/vali/test
        # 返回值为list_x和list_y，分别为list，list中每个元素为一个np格式的数组
        data, data_10 = self.data_split(target_use)
        N_ = data_10.shape[0]
        number_batch = int(np.ceil((N_-self.window_size-self.pred_len)/batch_size))
        return data, data_10, number_batch
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



class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask

class ProbAttn(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False, mask_flag=False):
        super(ProbAttn, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn



class ProbAttn1(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False, mask_flag=False):
        super(ProbAttn1, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top


    def _update_context(self, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)
        #print('a', attn.shape, V.shape)

        if self.output_attention:
            return (torch.matmul(attn, V), attn)
        else:
            return (torch.matmul(attn, V), None)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1).cumsum(-2)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        # update the context with selected top_k queries
        context, attn = self._update_context(values, scores_top, index, L_Q, attn_mask)
        #print('b', context.shape)
        
        return context, attn


class AttnLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttnLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, q, attn_mask=None):
        B, L, D = q.shape
        S = L
        H = self.n_heads

        queries = self.query_projection(q).view(B, L, H, -1)
        keys = self.key_projection(q).view(B, S, H, -1)
        values = self.value_projection(q).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, -1, D)

        return self.out_projection(out)
class LSTMAttn0(nn.Module):
    def __init__(self, time_len, scale, n_wt, n_feat, hidden_size, pred_len, filter_size=5, layer_num=2, n_heads=2, device='cpu'):
        super(LSTMAttn0, self).__init__()
        self.time_len, self.scale, self.n_wt, self.n_feat, self.hidden_size, self.layer_num = time_len, scale, n_wt, n_feat, hidden_size, layer_num

        self.attn = []
        self.attn.append(AttnLayer(ProbAttn1(factor=5), hidden_size, n_heads))
        self.attn.append(AttnLayer(ProbAttn1(factor=3), hidden_size, n_heads))
        self.attn = nn.ModuleList(self.attn)
        #self.pool = []
        #self.pool.append(nn.MaxPool1d(2, 2))
        #self.pool.append(nn.MaxPool1d(2, 2))
        #self.pool = nn.ModuleList(self.pool)

        self.conv = []
        self.conv.append(nn.Conv1d(n_feat+2, hidden_size, filter_size))
        self.conv.append(nn.Conv1d(hidden_size, hidden_size, filter_size))
        self.conv = nn.ModuleList(self.conv)
        self.layer_norm = nn.LayerNorm(12*hidden_size)
        self.lstm = nn.LSTM(input_size=12*hidden_size, num_layers=1, hidden_size=2*hidden_size, batch_first=True)
        self.final = nn.Linear(hidden_size*2, pred_len)
    def forward(self, wt, input, input2, device):
        batch_size = input.shape[0]
        h = input.view(batch_size*self.time_len, self.scale, self.n_wt, self.n_feat)[:, :, wt]
        PE1 = torch.sin(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len, self.scale, 1)
        PE2 = torch.cos(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len, self.scale, 1)
        h = torch.cat((h, PE1, PE2), dim=2).transpose(2, 1).contiguous()

        for i in range(self.layer_num):
            h = self.conv[i](h).transpose(2, 1).contiguous()
            h = self.attn[i](h)
            if (i < self.layer_num - 1):
                h = h.transpose(2, 1).contiguous()
        h = self.layer_norm(h.view(batch_size, self.time_len,12*self.hidden_size))
        _, (hn, __) = self.lstm(h)
        return self.final(hn[-1])
    def emb1(self, wt, input, input2, device):
        batch_size = input.shape[0]
        h = input.view(batch_size*self.time_len, self.scale, self.n_wt, self.n_feat)[:, :, wt]
        PE1 = torch.sin(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len, self.scale, 1)
        PE2 = torch.cos(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len, self.scale, 1)
        h = torch.cat((h, PE1, PE2), dim=2).transpose(2, 1).contiguous()

        for i in range(self.layer_num):
            h = self.conv[i](h).transpose(2, 1).contiguous()
            h = self.attn[i](h)
            if (i < self.layer_num - 1):
                h = h.transpose(2, 1).contiguous()
        h = self.layer_norm(h.view(batch_size, self.time_len,12*self.hidden_size))
        _, (hn, __) = self.lstm(h)
        return hn[-1]


class LSTMAttn(nn.Module):
    def __init__(self, time_len, scale, n_wt, n_feat, hidden_size, pred_len, filter_size=5, layer_num=2, n_heads=2, device='cpu'):
        super(LSTMAttn, self).__init__()
        self.time_len, self.scale, self.n_wt, self.n_feat, self.hidden_size, self.layer_num = time_len, scale, n_wt, n_feat, hidden_size, layer_num

        self.attn = []
        self.attn.append(AttnLayer(ProbAttn1(factor=5), hidden_size, n_heads))
        self.attn.append(AttnLayer(ProbAttn1(factor=3), hidden_size, n_heads))
        self.attn = nn.ModuleList(self.attn)
        self.conv = []
        self.conv.append(nn.Conv1d(n_feat, hidden_size, filter_size))
        self.conv.append(nn.Conv1d(hidden_size, hidden_size, filter_size))
        self.conv = nn.ModuleList(self.conv)
        self.layer_norm = nn.LayerNorm(12*hidden_size)
        self.lstm = nn.LSTM(input_size=12*hidden_size, num_layers=1, hidden_size=2*hidden_size, batch_first=True)
        self.final = nn.Linear(hidden_size*2, pred_len)
    def forward(self, wt, input, input2, device):
        batch_size = input.shape[0]
        h = input.view(batch_size*self.time_len, self.scale, self.n_wt, self.n_feat)
        for i in range(self.layer_num):
            h = self.conv[i](h).transpose(2, 1).contiguous()
            h = self.attn[i](h)
            if (i < self.layer_num - 1):
                h = h.transpose(2, 1).contiguous()
        h = self.layer_norm(h.view(batch_size, self.time_len, 12*self.hidden_size))
        _, (hn, __) = self.lstm(h)
        return self.final(hn[-1])
    def emb(self, wt, input, input2, device):
        batch_size = input.shape[0]
        h = input.view(batch_size*self.time_len, self.scale, self.n_wt, self.n_feat)[:, :, wt].transpose(2,1).contiguous()
        for i in range(self.layer_num):
            h = self.conv[i](h).transpose(2, 1).contiguous()
            h = self.attn[i](h)
            if (i < self.layer_num - 1):
                h = h.transpose(2, 1).contiguous()
        h = self.layer_norm(h.view(batch_size, self.time_len, 12*self.hidden_size))
        _, (hn, __) = self.lstm(h)
        return hn[-1]
    def emb1(self, wt, input, input2, device):
        batch_size = input.shape[0]
        h = input.view(batch_size*self.time_len, self.scale, self.n_wt, self.n_feat)[:, :, wt].transpose(2,1).contiguous()
        for i in range(self.layer_num):
            h = self.conv[i](h).transpose(2, 1).contiguous()
            h = self.attn[i](h)
            if (i < self.layer_num - 1):
                h = h.transpose(2, 1).contiguous()
        h = self.layer_norm(h.view(batch_size, self.time_len, 12*self.hidden_size))
        #_, (hn, __) = self.lstm(h)
        return h

class LSTMAttn2(nn.Module):
    def __init__(self, time_len, scale, n_wt, n_feat, hidden_size, pred_len, filter_size=5, layer_num=3, n_heads=2, device='cpu'):
        super(LSTMAttn2, self).__init__()
        self.time_len, self.scale, self.n_wt, self.n_feat, self.hidden_size, self.layer_num = time_len, scale, n_wt, n_feat, hidden_size, layer_num

        self.attn = []
        self.attn.append(AttnLayer(ProbAttn1(factor=5), hidden_size, n_heads))
        self.attn.append(AttnLayer(ProbAttn1(factor=3), hidden_size, n_heads))
        self.attn.append(AttnLayer(ProbAttn1(factor=1), hidden_size, n_heads))
        self.attn = nn.ModuleList(self.attn)
        self.conv = []
        self.conv.append(nn.Conv1d(n_feat, hidden_size, filter_size))
        self.conv.append(nn.Conv1d(hidden_size, hidden_size, filter_size))
        self.conv.append(nn.Conv1d(hidden_size, hidden_size, filter_size))
        self.conv = nn.ModuleList(self.conv)
        self.layer_norm = nn.LayerNorm(3*hidden_size)
        self.lstm = nn.LSTM(input_size=3*hidden_size, num_layers=1, hidden_size=hidden_size, batch_first=True)
        self.final = nn.Linear(hidden_size, pred_len)
    def forward(self, wt, input, input2, device):
        batch_size = input.shape[0]
        h = input.view(batch_size*self.time_len, self.scale, self.n_wt, self.n_feat)[:, :, wt].transpose(2,1).contiguous()
        for i in range(self.layer_num):
            h = self.conv[i](h).transpose(2, 1).contiguous()
            #print(i, h.shape)
            h = self.attn[i](h)
            if (i < self.layer_num - 1):
                h = h.transpose(2, 1).contiguous()
        h = self.layer_norm(h.view(batch_size, self.time_len, 3*self.hidden_size))
        _, (hn, __) = self.lstm(h)
        return self.final(hn[-1])
    def emb(self, wt, input, input2, device):
        batch_size = input.shape[0]
        h = input.view(batch_size*self.time_len, self.scale, self.n_wt, self.n_feat)[:, :, wt].transpose(2,1).contiguous()
        for i in range(self.layer_num):
            h = self.conv[i](h).transpose(2, 1).contiguous()
            h = self.attn[i](h)
            if (i < self.layer_num - 1):
                h = h.transpose(2, 1).contiguous()
        h = self.layer_norm(h.view(batch_size, self.time_len, 3*self.hidden_size))
        _, (hn, __) = self.lstm(h)
        return hn[-1]
    def emb1(self, wt, input, input2, device):
        batch_size = input.shape[0]
        h = input.view(batch_size*self.time_len, self.scale, self.n_wt, self.n_feat)[:, :, wt].transpose(2,1).contiguous()
        for i in range(self.layer_num):
            h = self.conv[i](h).transpose(2, 1).contiguous()
            h = self.attn[i](h)
            if (i < self.layer_num - 1):
                h = h.transpose(2, 1).contiguous()
        h = self.layer_norm(h.view(batch_size, self.time_len, 3*self.hidden_size))
        #_, (hn, __) = self.lstm(h)
        return h


class LSTMAttn3(nn.Module):
    def __init__(self, time_len, scale, n_wt, n_feat, hidden_size, pred_len, filter_size=5, layer_num=2, n_heads=2, device='cpu'):
        super(LSTMAttn3, self).__init__()
        self.time_len, self.scale, self.n_wt, self.n_feat, self.hidden_size, self.layer_num = time_len, scale, n_wt, n_feat, hidden_size, layer_num

        self.attn = []
        self.attn.append(nn.MaxPool1d(3, 3))
        self.attn.append(nn.MaxPool1d(2, 2))
        self.attn = nn.ModuleList(self.attn)
        self.conv = []
        self.conv.append(nn.Conv1d(n_feat+2, hidden_size, filter_size))
        self.conv.append(nn.Conv1d(hidden_size, hidden_size, filter_size))
        self.conv = nn.ModuleList(self.conv)
        self.layer_norm = nn.LayerNorm(hidden_size)
        #self.lstm = nn.LSTM(input_size=11*85*hidden_size, num_layers=1, hidden_size=2*hidden_size, batch_first=True)
        self.final = nn.Sequential(nn.Linear(hidden_size*847, hidden_size*11), nn.ReLU(), nn.Dropout(0.15), nn.Linear(hidden_size*11, pred_len))
    def forward(self, wt, input, input2, device):
        batch_size = input.shape[0]
        h = input.view(batch_size, self.time_len*self.scale, self.n_wt, self.n_feat)[:, :, wt]
        PE1 = torch.sin(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len,self.scale, 1).reshape(batch_size, self.time_len*self.scale, 1)
        PE2 = torch.cos(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len,self.scale, 1).reshape(batch_size, self.time_len*self.scale, 1)
        h = torch.cat((h, PE1, PE2), dim=2).transpose(2, 1).contiguous()

        for i in range(self.layer_num):
            h = self.conv[i](h)
            h = self.attn[i](h)

        h = h.transpose(2, 1).contiguous()
        h = self.layer_norm(h.view(batch_size, 847, self.hidden_size))
        #_, (hn, __) = self.lstm(h)
        return self.final(h.view(batch_size, -1))
    def emb1(self, wt, input, input2, device):
        batch_size = input.shape[0]
        h = input.view(batch_size*self.time_len, self.scale, self.n_wt, self.n_feat)[:, :, wt]
        PE1 = torch.sin(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len, self.scale, 1)
        PE2 = torch.cos(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len, self.scale, 1)
        h = torch.cat((h, PE1, PE2), dim=2).transpose(2, 1).contiguous()

        for i in range(self.layer_num):
            h = self.conv[i](h)
            h = self.attn[i](h)

        h = h.transpose(2, 1).contiguous()
        h = self.layer_norm(h.view(batch_size, self.time_len, 11*self.hidden_size))
        _, (hn, __) = self.lstm(h)
        return hn[-1]


class LSTMAttn4(nn.Module):
    def __init__(self, time_len, scale, n_wt, n_feat, hidden_size, pred_len, filter_size=5, layer_num=2, n_heads=2, device='cpu'):
        super(LSTMAttn4, self).__init__()
        self.time_len, self.scale, self.n_wt, self.n_feat, self.hidden_size, self.layer_num = time_len, scale, n_wt, n_feat, hidden_size, layer_num

        self.lstm = nn.LSTM(input_size=n_feat+2, num_layers=1, hidden_size=2*hidden_size, batch_first=True)
        self.final = nn.Linear(hidden_size*2, pred_len)
    def forward(self, wt, input, input2, device):
        batch_size = input.shape[0]
        h = input.view(batch_size, self.time_len*self.scale, self.n_wt, self.n_feat)[:, :, wt]
        PE1 = torch.sin(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len,self.scale, 1).reshape(batch_size, self.time_len*self.scale, 1)
        PE2 = torch.cos(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len,self.scale, 1).reshape(batch_size, self.time_len*self.scale, 1)
        h = torch.cat((h, PE1, PE2), dim=2).contiguous()

        _, (hn, __) = self.lstm(h)
        #_, (hn, __) = self.lstm(h)
        return self.final(hn[-1].view(batch_size, -1))
    def emb1(self, wt, input, input2, device):
        batch_size = input.shape[0]
        h = input.view(batch_size*self.time_len, self.scale, self.n_wt, self.n_feat)[:, :, wt]
        PE1 = torch.sin(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len, self.scale, 1)
        PE2 = torch.cos(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len, self.scale, 1)
        h = torch.cat((h, PE1, PE2), dim=2).transpose(2, 1).contiguous()

        for i in range(self.layer_num):
            h = self.conv[i](h)
            h = self.attn[i](h)

        h = h.transpose(2, 1).contiguous()
        h = self.layer_norm(h.view(batch_size, self.time_len, 11*self.hidden_size))
        _, (hn, __) = self.lstm(h)
        return hn[-1]


class LSTMAttn5(nn.Module):
    def __init__(self, time_len, scale, n_wt, n_feat, hidden_size, pred_len, filter_size=5, layer_num=2, n_heads=2, device='cpu'):
        super(LSTMAttn5, self).__init__()
        self.time_len, self.scale, self.n_wt, self.n_feat, self.hidden_size, self.layer_num = time_len, scale, n_wt, n_feat, hidden_size, layer_num

        self.attn = []
        self.attn.append(nn.MaxPool1d(2, 2))
        self.attn.append(nn.MaxPool1d(2, 2))
        self.attn = nn.ModuleList(self.attn)
        self.conv = []
        self.conv.append(nn.Conv1d(n_feat+2, hidden_size, filter_size))
        self.conv.append(nn.Conv1d(hidden_size, hidden_size, filter_size))
        self.conv = nn.ModuleList(self.conv)
        self.layer_norm = nn.LayerNorm(18*hidden_size)
        self.lstm = nn.LSTM(input_size=18*hidden_size, num_layers=1, hidden_size=2*hidden_size, batch_first=True)
        self.final = nn.Linear(hidden_size*2, pred_len)
    def forward(self, wt, input, input2, device):

        batch_size = input.shape[0]
        h = input.view(batch_size*self.time_len, self.scale, self.n_wt, self.n_feat)[:, :, wt]
        PE1 = torch.sin(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len, self.scale, 1)
        PE2 = torch.cos(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len, self.scale, 1)
        h = torch.cat((h, PE1, PE2), dim=2).transpose(2, 1).contiguous()

        for i in range(self.layer_num):
            h = self.conv[i](h)
            h = self.attn[i](h)

        h = h.transpose(2, 1).contiguous()
        h = self.layer_norm(h.view(batch_size, self.time_len, 18*self.hidden_size))
        _, (hn, __) = self.lstm(h)
        return self.final(hn[-1])
class LSTMAttn6(nn.Module):
    def __init__(self, time_len, scale, n_wt, n_feat, hidden_size, pred_len, filter_size=5, layer_num=2, n_heads=2, device='cpu'):
        super(LSTMAttn6, self).__init__()
        self.time_len, self.scale, self.n_wt, self.n_feat, self.hidden_size, self.layer_num = time_len, scale, n_wt, n_feat, hidden_size, layer_num

        self.attn = []
        self.attn.append(AttnLayer(ProbAttn1(factor=5), hidden_size, n_heads))
        self.attn.append(AttnLayer(ProbAttn1(factor=3), hidden_size, n_heads))
        self.attn = nn.ModuleList(self.attn)
        self.project_in = nn.Sequential(nn.Linear(n_feat+2, hidden_size), nn.Dropout(0.15), nn.ReLU(), nn.Linear(hidden_size, hidden_size))

        self.layer_norm = nn.LayerNorm(12*hidden_size)
        self.lstm = nn.LSTM(input_size=12*hidden_size, num_layers=1, hidden_size=2*hidden_size, batch_first=True)
        self.final = nn.Linear(hidden_size*2, pred_len)
    def forward(self, wt, input, input2, device):
        batch_size = input.shape[0]
        h = input.view(batch_size*self.time_len, self.scale, self.n_wt, self.n_feat)[:, :, wt]
        h = input.view(batch_size*self.time_len, self.scale, self.n_wt, self.n_feat)[:, :, wt]
        PE1 = torch.sin(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len, self.scale, 1)
        PE2 = torch.cos(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len, self.scale, 1)
        h = torch.cat((h, PE1, PE2), dim=2)

        h = self.project_in(h)
        for i in range(self.layer_num):
            h = self.attn[i](h)
        h = self.layer_norm(h.view(batch_size, self.time_len,12*self.hidden_size))
        _, (hn, __) = self.lstm(h)
        return self.final(hn[-1])

class LSTMAttn7(nn.Module):
    def __init__(self, time_len, scale, n_wt, n_feat, hidden_size, pred_len, filter_size=5, layer_num=2, n_heads=2, device='cpu'):
        super(LSTMAttn7, self).__init__()
        self.time_len, self.scale, self.n_wt, self.n_feat, self.hidden_size, self.layer_num = time_len, scale, n_wt, n_feat, hidden_size, layer_num
        self.attn = []
        self.attn.append(AttnLayer(ProbAttn1(factor=5), hidden_size, n_heads))
        self.attn.append(AttnLayer(ProbAttn1(factor=3), hidden_size, n_heads))
        self.attn = nn.ModuleList(self.attn)
        self.conv = []
        self.conv.append(nn.Conv1d(n_feat+2, hidden_size, filter_size))
        self.conv.append(nn.Conv1d(hidden_size, hidden_size, filter_size))
        self.conv = nn.ModuleList(self.conv)
        self.layer_norm = nn.LayerNorm(12*hidden_size)
        #self.lstm = nn.LSTM(input_size=12*hidden_size, num_layers=1, hidden_size=2*hidden_size, batch_first=True)
        self.final = nn.Sequential(nn.Linear(hidden_size*12*time_len, hidden_size * 12), nn.Dropout(0.15), nn.ReLU(), nn.Linear(hidden_size*12, pred_len))
    def forward(self, wt, input, input2, device):
        batch_size = input.shape[0]
        h = input.view(batch_size*self.time_len, self.scale, self.n_wt, self.n_feat)[:, :, wt]
        PE1 = torch.sin(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len, self.scale, 1)
        PE2 = torch.cos(torch.arange(self.scale, device=device)/self.scale)[None, :, None].expand(batch_size*self.time_len, self.scale, 1)
        h = torch.cat((h, PE1, PE2), dim=2).transpose(2, 1).contiguous()
        for i in range(self.layer_num):
            h = self.conv[i](h).transpose(2, 1).contiguous()
            h = self.attn[i](h)
            if (i < self.layer_num - 1):
                h = h.transpose(2, 1).contiguous()
        h = self.layer_norm(h.view(batch_size, self.time_len, 12*self.hidden_size))
        return self.final(h.view(batch_size, -1))


class ConvAttn2(nn.Module):
    def __init__(self, time_len, scale, n_wt, n_feat, hidden_size, pred_len, filter_size=4, layer_num=2, n_heads=2, device='cpu'):
        super(ConvAttn2, self).__init__()
        self.cnn_model = []
        self.cnn2 = nn.Conv2d(hidden_size, hidden_size, filter_size)
        self.mhas = []
        self.mhas2 = []

        h, w = time_len, scale
        self.time_len, self.scale, self.n_wt, self.n_feat, self.hidden_size, self.layer_num, self.filter_size = time_len, scale, n_wt, n_feat, hidden_size, layer_num, filter_size
        for i in range(self.layer_num):
            if i == 0:
                self.cnn_model.append(nn.Conv2d(n_feat, hidden_size, filter_size))
            else:
                self.cnn_model.append(nn.Conv2d(hidden_size, hidden_size, filter_size))

            h = h - filter_size + 1
            w = w - filter_size + 1
            self.cnn_model.append(nn.MaxPool2d(3, 3))
            h = int((h - 3)/3+1)
            w = int((w - 3)/3+1)

        for i in range(self.layer_num):
            self.mhas.append(AttnLayer(ProbAttn(), hidden_size, n_heads))
            self.mhas2.append(MHA(n_heads, hidden_size, hidden_size))
        self.cnn_model = nn.ModuleList(self.cnn_model)
        self.mhas = nn.ModuleList(self.mhas)
        h = h - 3
        w = w - 3
        self.mhas2 = nn.ModuleList(self.mhas)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lstm = nn.LSTM(input_size=n_feat, num_layers=1, hidden_size=hidden_size, batch_first=True)

        self.project = nn.Sequential(nn.Linear(h * w * hidden_size, hidden_size), nn.Dropout(0.15), nn.ReLU(), nn.Linear(hidden_size, hidden_size))

        self.final = nn.Linear(hidden_size*2, pred_len)
    def forward(self, wt, input, input2, device):
        batch_size = input.shape[0]
        h = input.permute(0, 3, 4, 1, 2).contiguous().view(batch_size*self.n_wt, self.n_feat, self.time_len, self.scale)
        a, b = self.time_len, self.scale
        for i in range(self.layer_num):
            h = self.cnn_model[2*i](h)
            h = self.cnn_model[2*i+1](h)
            a = a - self.filter_size + 1
            b = b - self.filter_size + 1
            a = int((a - 3)/3+1)
            b = int((b - 3)/3+1)
        h = h.view(batch_size, self.n_wt, self.hidden_size, a, b).permute(0, 3, 4, 1, 2).contiguous().view(batch_size, a*b*self.n_wt, self.hidden_size)
        for i in range(self.layer_num):
            h = self.mhas[i](h)
        h = self.layer_norm(h.view(batch_size, a, b, self.n_wt, self.hidden_size))
        h = h.permute(0, 3, 4, 1, 2).contiguous().view(batch_size*self.n_wt, self.hidden_size, a, b)
        h = self.cnn2(h)
        a -= 3
        b -= 3
        h = h.view(batch_size, self.n_wt, self.hidden_size, a, b).permute(0, 3, 4, 1, 2).contiguous().view(batch_size, a*b*self.n_wt, self.hidden_size)
        for i in range(self.layer_num):
            h = self.mhas2[i](h)
        h = self.layer_norm(h.view(batch_size, a*b, self.n_wt, self.hidden_size))
        x, (hn, cn) = self.lstm(input2)
        out = self.project(h[:, :, wt, :].reshape(batch_size, -1))
        return self.final(torch.cat((out, hn[-1]), dim=-1))

class ConvAttn3(nn.Module):
    def __init__(self, time_len, scale, n_wt, n_feat, hidden_size, pred_len, filter_size=4, layer_num=2, n_heads=2, device='cpu'):
        super(ConvAttn3, self).__init__()
        self.cnn_model = []
        #self.cnn2 = nn.Conv2d(hidden_size, hidden_size, filter_size)
        self.mhas = []
        self.mhas2 = []

        h, w = time_len, scale
        self.time_len, self.scale, self.n_wt, self.n_feat, self.hidden_size, self.layer_num, self.filter_size = time_len, scale, n_wt, n_feat, hidden_size, layer_num, filter_size
        for i in range(self.layer_num):
            if i == 0:
                self.cnn_model.append(nn.Conv2d(n_feat, hidden_size, filter_size))
            else:
                self.cnn_model.append(nn.Conv2d(hidden_size, hidden_size, filter_size))

            h = h - filter_size + 1
            w = w - filter_size + 1
            self.cnn_model.append(nn.MaxPool2d(4, 4))
            h = int((h - 4)/4+1)
            w = int((w - 4)/4+1)

        for i in range(self.layer_num):
            self.mhas.append(AttnLayer(ProbAttn(), hidden_size, n_heads))
            self.mhas2.append(MHA(n_heads, hidden_size, hidden_size))
        self.cnn_model = nn.ModuleList(self.cnn_model)
        self.mhas = nn.ModuleList(self.mhas)
        #h = h - 3
        #w = w - 3
        self.mhas2 = nn.ModuleList(self.mhas)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lstm = nn.LSTM(input_size=n_feat, num_layers=1, hidden_size=hidden_size, batch_first=True)

        self.project = nn.Sequential(nn.Linear(h * w * hidden_size, hidden_size), nn.Dropout(0.15), nn.ReLU(), nn.Linear(hidden_size, hidden_size))

        self.final = nn.Linear(hidden_size*2, pred_len)
    def forward(self, wt, input, input2, device):
        batch_size = input.shape[0]

        a, b = self.time_len, self.scale
        h = input.view(batch_size, a, b, self.n_wt, -1).permute(0, 3, 4, 1, 2).contiguous().view(batch_size*self.n_wt, -1, a, b)

        for i in range(self.layer_num):
            h = self.cnn_model[2*i](h)
            h = self.cnn_model[2*i+1](h)
            a = a - self.filter_size + 1
            b = b - self.filter_size + 1
            a = int((a - 4)/4+1)
            b = int((b - 4)/4+1)
            h = h.view(batch_size, self.n_wt, self.hidden_size, a, b).permute(0, 3, 4, 1, 2).contiguous().view(batch_size, a*b*self.n_wt, self.hidden_size)
            h = self.mhas[i](h)
            h = self.layer_norm(h)
            if (i < self.layer_num-1):
                h = h.view(batch_size, a, b, self.n_wt, self.hidden_size).permute(0, 3, 4, 1, 2).contiguous().view(batch_size*self.n_wt, self.hidden_size, a, b)



        #h = h.permute(0, 3, 4, 1, 2).contiguous().view(batch_size*self.n_wt, self.hidden_size, a, b)
        #h = self.cnn2(h)
        #a -= 3
        #b -= 3
        #h = h.view(batch_size, self.n_wt, self.hidden_size, a, b).permute(0, 3, 4, 1, 2).contiguous().view(batch_size, a*b*self.n_wt, self.hidden_size)
        for i in range(self.layer_num):
            h = self.mhas2[i](h)
        #h = self.layer_norm(h.view(batch_size, a*b, self.n_wt, self.hidden_size))
        h = h.view(batch_size, a*b, self.n_wt, self.hidden_size)
        x, (hn, cn) = self.lstm(input2)
        out = self.project(h[:, :, wt, :].reshape(batch_size, -1))
        return self.final(torch.cat((out, hn[-1]), dim=-1))






class LSTM(nn.Module):
    def __init__(self, time_len, n_feat, hidden_size, pred_len=6, device='cpu'):
        super(LSTM, self).__init__()

        self.time_len, self.n_feat, self.hidden_size = time_len, n_feat, hidden_size

        self.lstm = nn.LSTM(input_size=n_feat, num_layers=1, hidden_size=hidden_size, batch_first=True)
        self.final = nn.Linear(hidden_size, pred_len)
    def forward(self, input2, device):
        x, (hn, cn) = self.lstm(input2)
        return self.final(hn[-1])
    def emb(self, input2, device):
        x, (hn, cn) = self.lstm(input2)
        return hn[-1]



class LSTMComb(nn.Module):
    def __init__(self, n_feat, hidden_size, pred_len, device='cpu'):
        super(LSTMComb, self).__init__()
        self.project1 = nn.Sequential(nn.Linear(hidden_size*12, hidden_size*3), nn.Dropout(0.15), nn.ReLU(), nn.Linear(hidden_size*3, hidden_size))
        self.project2 = nn.Sequential(nn.Linear(n_feat, hidden_size), nn.Dropout(0.15), nn.ReLU(), nn.Linear(hidden_size, hidden_size))

        self.lstm = nn.LSTM(input_size=hidden_size*2, num_layers=1, hidden_size=hidden_size*2, batch_first=True)

        self.final = nn.Linear(hidden_size*2, pred_len)
    def forward(self, input1, input2, device):
        batch_size = input2.shape[0]
        input1 = self.project1(input1)
        input2 = self.project2(input2)
        x, (hn, cn) = self.lstm(torch.cat((input1, input2), axis=-1))
        return self.final(hn[-1])

class LSTMSingle(nn.Module):
    def __init__(self, n_feat, hidden_size, pred_len, device='cpu'):
        super(LSTMSingle, self).__init__()

        self.project2 = nn.Sequential(nn.Linear(n_feat, hidden_size*2), nn.Dropout(0.15), nn.ReLU(), nn.Linear(hidden_size*2, hidden_size*2))
        self.lstm = nn.LSTM(input_size=hidden_size*2, num_layers=1, hidden_size=hidden_size*2, batch_first=True)

        self.final = nn.Linear(hidden_size*2, pred_len)
    def forward(self, input2, device):
        batch_size = input2.shape[0]
        input2 = self.project2(input2)
        x, (hn, cn) = self.lstm(input2)
        return self.final(hn[-1])
    def emb(self, input2, device):
        batch_size = input2.shape[0]
        input2 = self.project2(input2)
        x, (hn, cn) = self.lstm(input2)
        return hn[-1]


class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, pred_len, device='cpu'):
        super(DNN, self).__init__()
        self.project_out = nn.Sequential(nn.Linear(input_size, hidden_size*2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size*2, hidden_size), nn.ReLU())
        self.final = nn.Linear(hidden_size, pred_len)
    def forward(self, input, device):
        embed = self.project_out(input.view(input.size(0), -1))
        return self.final(embed)
    def emb(self, input, device):
        return self.project_out(input.view(input.size(0), -1))

class GLSTM(nn.Module):
    def __init__(self, time_len, scale, n_wt, n_feat, hidden_size, pred_len, filter_size=5, layer_num=2, n_heads=2, device='cpu'):
        super(GLSTM, self).__init__()
        self.time_len, self.scale, self.n_wt, self.n_feat, self.hidden_size, self.layer_num = time_len, scale, n_wt, n_feat, hidden_size, layer_num
        self.lstm = nn.LSTM(input_size=n_feat, num_layers=1, hidden_size=hidden_size, batch_first=True)
        self.final = nn.Linear(hidden_size, pred_len)
    def forward(self, wt, input, input2, device):
        batch_size = input2.shape[0]
        h = input2.view(batch_size, self.time_len, self.n_feat)
        x, (hn, cn) = self.lstm(h)
        return self.final(hn[-1].view(batch_size, -1))

class GRU(nn.Module):
    def __init__(self, time_len, scale, n_wt, n_feat, hidden_size, pred_len, filter_size=5, layer_num=2, n_heads=2, device='cpu'):
        super(GRU, self).__init__()
        self.time_len, self.scale, self.n_wt, self.n_feat, self.hidden_size, self.layer_num = time_len, scale, n_wt, n_feat, hidden_size, layer_num
        self.lstm = nn.GRU(input_size=n_feat, num_layers=1, hidden_size=hidden_size, batch_first=True)
        self.final = nn.Linear(hidden_size, pred_len)
    def forward(self, wt, input, input2, device):
        batch_size = input2.shape[0]
        h = input2.view(batch_size, self.time_len, self.n_feat)
        x, hn = self.lstm(h)
        return self.final(hn[-1].view(batch_size, -1))


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class CNN(nn.Module):
    def __init__(self, input_shape, device='cpu'):
        super(CNN, self).__init__()
        self.filter_size = 4
        self.channel_in = input_shape[0]*input_shape[1]
        self.input_h = input_shape[2]
        self.fc_dim = 16
        self.layer_num = 1
        self.channel_num = 64
        self.device = device
        self.cnn_model = []
        # Input shape: batch_size, physical, wtnumber, laglength
        self.Pool_func = nn.MaxPool1d
        h = self.input_h
        self.activ_func = nn.ReLU()
        for i in range(self.layer_num):
            h = h - self.filter_size + 1
            h = int((h - 2)/2+1)
        if (h > 0):
            self.model_valid = 1
            h = self.input_h
            for i in range(self.layer_num):
                if i == 0:
                    self.cnn_model.append(nn.Conv1d(self.channel_in, self.channel_num, self.filter_size))
                else:
                    self.cnn_model.append(nn.Conv1d(self.channel_num, self.channel_num, self.filter_size))
                h = h - self.filter_size + 1
                self.cnn_model.append(self.Pool_func(2))
                h = int((h - 2)/2+1)
            self.mlp_model = nn.Sequential(
                nn.Linear(h * self.channel_num, self.fc_dim),
                nn.Dropout(0.1),
                self.activ_func,
                nn.Linear(self.fc_dim, 6))
            self.cnn_model = ListModule(*self.cnn_model)
        else:
            self.model_valid = 0
    def forward(self, x_input, device):
        batch_size = x_input.shape[0]
        h = x_input.transpose(2, 1).contiguous()
        for i in range(self.layer_num):
            h = self.cnn_model[2*i](h)
            h = self.cnn_model[2*i+1](h)
        h = h.view(batch_size, -1)
        out = self.mlp_model(h)
        return out
    def emb(self, x_input, device):
        batch_size = x_input.shape[0]
        h = x_input.transpose(2, 1).contiguous()
        for i in range(self.layer_num):
            # CNN部分
            h = self.cnn_model[2*i](h)
            h = self.cnn_model[2*i+1](h)
        h = h.view(batch_size, -1)
        for i in range(len(self.mlp_model)-1):
            h = self.mlp_model[i](h)
        return h

def data_save(data, save_path):
    data = pd.DataFrame(data)
    data.to_csv(save_path, index=None, header=None)
def RMSE_compute(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2, axis=0))
def MAPE_compute(pred, true):
    return np.mean(np.abs(pred-true)/true, axis=0)
