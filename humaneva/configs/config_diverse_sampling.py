#!/usr/bin/env python
# encoding: utf-8
'''
@project : t2_twogroupbase20_epo180_div15_ade37
@file    : config_decoupled.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-03-04 09:48
'''

import getpass
import os
import numpy as np


class ConfigDiverseSampling():
    def __init__(self, exp_name="", device="cuda:0", num_works=0):
        self.platform = getpass.getuser()
        self.exp_name = exp_name

        self.class_num = 5

        # >>> model

        self.z_dim = 64
        self.hidden_dim = 256
        self.base_dim = 128
        self.base_num_p1 = 40

        # >>> data
        self.t_his = 15
        self.t_pred = 60
        self.dct_n = 10
        self.t_total = self.t_his + self.t_pred

        self.sample_step_train = 1
        self.sample_step_test = self.t_his

        self.sub_len_train = 2000
        self.multimodal_threshold = 0.5
        self.train_similar_cnt = 10

        self.subjects = {'train': ['Train/S1', 'Train/S2', 'Train/S3'],
                         'test': ['Validate/S1', 'Validate/S2', 'Validate/S3']}
        self.joint_used = [i for i in range(15)]  # 15 个点
        self.parents = [-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1]  # 15点

        self.I17_plot = [0, 8, 9, 0, 11, 12, 0, 1, 1, 2, 3, 1, 5, 6]
        self.J17_plot = [8, 9, 10, 11, 12, 13, 1, 14, 2, 3, 4, 5, 6, 7]
        self.LR17_plot = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]

        self.I16_plot = [0, 7, 8,  0, 10, 11,  0, 0, 1, 2, 0, 4, 5]
        self.J16_plot = [7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 4, 5, 6]
        self.parents_16 = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0]  # 15点

        self.LR16_plot = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]

        self.node_n = len(self.joint_used) - 1

        # >>> runner
        self.nk = 50
        self.seperate_head = 25

        self.lr_t2 = 1e-3
        self.train_batch_size = 16
        self.test_batch_size = 1
        self.epoch_t2 = 1500
        self.epoch_fix_t2 = 100

        self.temperature_p1 = 0.85 # 1
        self.minthreshold = 20
        self.dlow_scale = 50

        # todo: dlow t2 > kl:ade:div = 1:2:50(20);
        #  gsps t2 > divp1 : divp2 : ade : mmade : recovhis : limblen : angle : lkh = 5(15) : 10(50) : 2 : 1, 100: 500 : 100 : 0.01
        self.t2_kl_p1_weight = 0.1 # 0.2
        self.t2_ade_weight = 25 # 10
        self.t2_diversity_weight = 100 # 30, ablation hinge v.s. energy 时，energy 这一权重设置了 100 / 300

        self.dropout_rate = 0

        self.device = device
        self.num_works = num_works

        self.ckpt_dir = os.path.join("./ckpt/", exp_name)
        if not os.path.exists(os.path.join(self.ckpt_dir, "models")):
            os.makedirs(os.path.join(self.ckpt_dir, "models"))
        if not os.path.exists(os.path.join(self.ckpt_dir, "images")):
            os.makedirs(os.path.join(self.ckpt_dir, "images"))

        if self.platform == "Drolab":
            self.base_data_dir = os.path.join(r"./dataset")
        else:
            self.base_data_dir = os.path.join(r"./dataset")

        self.valid_angle_path = os.path.join(self.base_data_dir, "humaneva_valid_angle.p")
        # >>> gsps
        self.similar_idx_path = os.path.join(self.base_data_dir, "humaneva_multi_modal", "t_his15_1_thre0.500_t_pred60_thre0.010_index_filterd.npz")
        self.similar_pool_path = os.path.join(self.base_data_dir, "humaneva_multi_modal", "data_candi_t_his15_t_pred60_skiprate15.npz")
        self.model_path_t1 = os.path.join(r"./ckpt/pretrained/humaneva_t1.pth")


