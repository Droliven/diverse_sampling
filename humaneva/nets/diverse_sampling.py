#!/usr/bin/env python
# encoding: utf-8
'''
@project : baseresample_likegsps
@file    : decoupled.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-13 10:57
'''

import torch
from torch.nn import Module, Sequential, ModuleList, ModuleDict, Linear, GELU, Tanh, BatchNorm1d
import numpy as np

from .gcn_layers import GraphConv, GraphConvBlock, ResGCB


class DiverseSampling(Module):
    def __init__(self, node_n=16, hidden_dim=256, base_dim = 64, z_dim=64, dct_n=10, base_num_p1=10, dropout_rate=0):
        super(DiverseSampling, self).__init__()
        self.z_dim = z_dim
        self.base_dim = base_dim
        self.base_num_p1 = base_num_p1

        self.condition_enc = Sequential(
            GraphConvBlock(in_len=3*dct_n, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=False),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(in_len=hidden_dim, out_len=hidden_dim, in_node_n=node_n, out_node_n=node_n, dropout_rate=dropout_rate, bias=True, residual=True),
        )
        self.bases_p1 = Sequential(
            Linear(node_n*hidden_dim, self.base_num_p1 * self.base_dim),  # 42*256, 20*128
            BatchNorm1d(self.base_num_p1 * self.base_dim),
            Tanh()
        )  # 10.4M

        self.mean_p1 = Sequential(
            Linear(self.base_dim, 64),  # 48*128, (10+10), 256
            BatchNorm1d(64),
            Tanh(),
            Linear(64, self.z_dim)
        )
        self.logvar_p1 = Sequential(
            Linear(self.base_dim, 64),  # 48*128, (10+10), 256
            BatchNorm1d(64),
            Tanh(),
            Linear(64, self.z_dim)
        )


    def forward(self, condition, repeated_eps=None, many_weights=None, multi_modal_head=10):
        '''

        Args:
            condition: [b, 48, 25] / [b, 16, 3*10]
            repeated_eps: b*50, 64
        Returns:

        '''
        b, v, ct = condition.shape
        condition_enced = self.condition_enc(condition)  # b, 16, 256

        bases = self.bases_p1(condition_enced.view(b, -1)).view(b, self.base_num_p1, self.base_dim)  # b, 10, 64


        repeat_many_bases = torch.repeat_interleave(bases, repeats=multi_modal_head, dim=0)  # b*h, 10, 64
        many_bases_blending = torch.matmul(many_weights, repeat_many_bases).squeeze(dim=1).view(-1, self.base_dim)  # b*h, 64

        all_mean = self.mean_p1(many_bases_blending)
        all_logvar = self.logvar_p1(many_bases_blending)


        all_z = torch.exp(0.5 * all_logvar) * repeated_eps + all_mean

        return all_z, all_mean, all_logvar




if __name__ == '__main__':
    m = DiverseSampling(node_n=16, hidden_dim=256, base_dim = 128, z_dim=64, dct_n=10, base_num_p1=20, dropout_rate=0).cuda()
    print(f"{sum(p.numel() for p in m.parameters()) / 1e6}")

    pass



