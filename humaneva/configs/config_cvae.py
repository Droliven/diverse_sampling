#!/usr/bin/env python
# encoding: utf-8
'''
@project : t2_twogroupbase20_epo180_div15_ade37
@file    : config_cvae.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-03-04 10:02
'''

import getpass
import os
import numpy as np


class ConfigCVAE():
    def __init__(self, exp_name="", device="cuda:0", num_works=0):
        self.platform = getpass.getuser()
        self.exp_name = exp_name

        # >>> model
        self.nk = 50
        self.seperate_head = 25

        self.z_dim = 64
        self.hidden_dim = 256

        # >>> data
        self.t_his = 15
        self.t_pred = 60
        self.dct_n = 10
        self.t_total = self.t_his + self.t_pred

        self.sample_step_train = 1
        self.sample_step_test = self.t_his

        self.sub_len_train = 2000
        self.multimodal_threshold = 0.5

        self.subjects = {'train': ['Train/S1', 'Train/S2', 'Train/S3'],
                         'test': ['Validate/S1', 'Validate/S2', 'Validate/S3']}
        self.joint_used = [i for i in range(15)]  # 15 个点
        self.parents = [-1, 0, 1, 2, 3, 1, 5, 6, 0, 8, 9, 0, 11, 12, 1]  # 15点
        self.I17_plot = [0, 8, 9, 0, 11, 12, 0, 1, 1, 2, 3, 1, 5, 6]
        self.J17_plot = [8, 9, 10, 11, 12, 13, 1, 14, 2, 3, 4, 5, 6, 7]
        self.LR17_plot = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
        self.I16_plot = [0, 7, 8, 0, 10, 11, 0, 0, 1, 2, 0, 4, 5]
        self.J16_plot = [7, 8, 9, 10, 11, 12, 13, 1, 2, 3, 4, 5, 6]
        self.LR16_plot = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
        self.node_n = len(self.joint_used) - 1

        # >>> runner
        self.lr_t1 = 1e-3
        self.train_batch_size = 32
        self.test_batch_size = 1
        self.epoch_t1 = 2000
        self.epoch_fix_t1 = 150

        self.t1_recons_weight = 1
        self.t1_kl_weight = 0.05
        self.t1_vec_weight = 1000
        self.t1_recoverhis_weight = 100
        self.t1_limblen_weight = 1000
        self.t1_angle_weight = 100

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


