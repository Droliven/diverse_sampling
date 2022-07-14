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
        self.dct_n = 10
        self.hidden_dim = 256
        self.node_n = 16

        # >>> data
        self.t_his = 25
        self.t_pred = 100
        self.t_total = self.t_his + self.t_pred

        self.sample_step_train = 1
        self.sample_step_test = self.t_his

        self.sub_len_train = 5000
        self.multimodal_threshold = 0.5
        self.train_similar_cnt = 0

        self.subjects = {"train": [f"S{i}" for i in [1, 5, 6, 7, 8]], "test": [f"S{i}" for i in [9, 11]]}
        self.joint_used=[0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
        self.parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        self.I32_plot = [0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12, 16, 17, 18, 19, 20, 19, 22, 12, 24, 25,
                         26, 27,
                         28,
                         27, 30]
        self.J32_plot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                         27, 28,
                         29,
                         30, 31]
        self.LR32_plot = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

        self.I17_plot = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
        self.J17_plot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.LR17_plot = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]

        # >>> runner
        self.lr_t1 = 1e-3
        self.train_batch_size = 64
        self.test_batch_size = 1
        self.epoch_t1 = 2000
        self.epoch_fix_t1 = 100

        self.t1_recons_weight = 1
        self.t1_kl_weight = 0.5  # 这里绝对不能设为 0.1，否则准确性太差，0.5 和 1 可以设置
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

        self.valid_angle_path = os.path.join(self.base_data_dir, "h36m_valid_angle.p")
        # >>> gsps
        self.similar_idx_path = os.path.join(self.base_data_dir, "data_multi_modal", "t_his25_1_thre0.500_t_pred100_thre0.100_filtered_dlow.npz")
        self.similar_pool_path = os.path.join(self.base_data_dir, "data_multi_modal", "data_candi_t_his25_t_pred100_skiprate20.npz")

