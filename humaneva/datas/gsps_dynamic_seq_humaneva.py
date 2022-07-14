#!/usr/bin/env python
# encoding: utf-8
'''
@project : dlow_reimp
@file    : gsps_dynamic_seq_humaneva.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-02-27 15:59
'''

import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
import torch

class MaoweiGSPS_Dynamic_Seq_Humaneva():
    def __init__(self, t_his=25, t_pred=100, similar_cnt=10, dynamic_sub_len=5000, batch_size=8,
                 data_path=r"./dataset",
                 similar_idx_path=r"./dataset/data_multi_modal/t_his25_1_thre0.500_t_pred100_thre0.100_filtered_dlow.npz",
                 similar_pool_path=r"./dataset//data_multi_modal/data_candi_t_his25_t_pred100_skiprate20.npz",
                 subjects={"train": [f"S{i}" for i in [1, 5, 6, 7, 8]], "test": [f"S{i}" for i in [9, 11]]},
                 joint_used_17=[0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27],
                 parents_17=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15], mode="train",
                 multimodal_threshold=0.5, is_debug=False):

        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.similar_cnt = similar_cnt
        self.dynamic_sub_len = dynamic_sub_len
        self.batch_size = batch_size
        self.subjects = subjects[mode]
        self.mode = mode
        self.multimodal_threshold = multimodal_threshold
        self.is_debug = is_debug

        self.actions = "all"

        self.parents_17 = parents_17

        data_o = np.load(os.path.join(data_path, "data_3d_humaneva15.npz"), allow_pickle=True)['positions_3d'].item()
        self.data = dict(
            filter(lambda x: x[0] in self.subjects, data_o.items()))  # dict [n, 125, 17, 3], hip 不为 0，但其他点会对hip相对化

        if self.mode == 'train':
            self.data['Train/S3'].pop('Walking 1 chunk0')
            self.data['Train/S3'].pop('Walking 1 chunk2')
        else:
            self.data['Validate/S3'].pop('Walking 1 chunk4')

        if mode == "train" and similar_cnt > 0:
            self.data_similar_idx = np.load(similar_idx_path, allow_pickle=True)['data_multimodal'].item()  # dict

            similar_pool = np.load(similar_pool_path, allow_pickle=True)[
                'data_candidate.npy']  # [18627, 125, 16, 3], 其实是从整个数据集上面抽取出来的 18627 个样本，他们包含 16 个点的运动序列，用单位向量表示，消除个体差异
            self.data_simar_pool = {}  # dict [18627, 125, 17, 3] hip 统统为 0

        for key in list(self.data.keys()):
            self.data[key] = dict(filter(lambda x: (self.actions == 'all' or
                                                 all([a in x[0] for a in self.actions]))
                                                and x[1].shape[0] >= self.t_total, self.data[key].items()))
            if len(self.data[key]) == 0:
                self.data.pop(key)

        for sub in self.data.keys():
            for action in self.data[sub].keys():
                seq = self.data[sub][action][:, joint_used_17, :]  # n, 17, 3
                seq[:, 1:] -= seq[:, :1]  # 相对化
                self.data[sub][action] = seq

                if mode == "train" and similar_cnt > 0 and (sub not in self.data_simar_pool.keys()):
                    x0 = np.copy(seq[None, :1, ...])  # 1, 1, 17, 3 序列第一帧
                    x0[:, :, 0] = 0  # 第一帧的第一个关节点
                    self.data_simar_pool[sub] = self.normalized_vector_to_adpative_coordinate(similar_pool, x0=x0)

        print(
            f"{mode} Data Loaded, dynamic_sub_len: {dynamic_sub_len}, similar_cnt: {similar_cnt}, batch_size: {batch_size}!")

    def sample(self):
        while True:
            subject = np.random.choice(self.subjects)
            dict_s = self.data[subject]
            action = np.random.choice(list(dict_s.keys()))
            seq = dict_s[action]
            if seq.shape[0] > self.t_total:
                break

        frame_start = np.random.randint(seq.shape[0] - self.t_total)
        frame_end = frame_start + self.t_total
        data = seq[frame_start: frame_end]  # 125, 17, 3
        if self.mode == "train" and self.similar_cnt > 0 and (subject in self.data_similar_idx.keys()):
            pool = self.data_simar_pool[subject]
            idx_multi = self.data_similar_idx[subject][action][frame_start]
            data_similar = pool[idx_multi]  # [n, 125, 17, 3]

            if len(data_similar) > 0:
                data_similar[:, :self.t_his] = data[None, ...][:, :self.t_his]  # 用原来的拼接
                if data_similar.shape[0] > self.similar_cnt:
                    st0 = np.random.get_state()
                    remain_idx = np.random.choice(np.arange(data_similar.shape[0]), self.similar_cnt, replace=False)
                    data_similar = data_similar[remain_idx]  # 10, 125, 17, 3
                    np.random.set_state(st0)  # todo 这又是在干嘛

            data_similar = np.concatenate(
                [data_similar, np.zeros_like(data[None, ...][[0] * (self.similar_cnt - data_similar.shape[0])])], axis=0)  # repeat

            data = data[:, 1:, :].reshape(1, self.t_total, -1).transpose(0, 2, 1)  # 1, 48, 125
            data_similar = data_similar[:, :, 1:, :].reshape(1, self.similar_cnt, self.t_total, -1).transpose(0, 1, 3,
                                                                                                              2)  # 1, 10, 48, 125
            return data, data_similar
        else:
            data = data[:, 1:, :].reshape(1, self.t_total, -1).transpose(0, 2, 1)  # 1, 48, 125
            return data

    def batch_generator(self):
        if self.is_debug:
            self.dynamic_sub_len = 200

        for i in range(self.dynamic_sub_len // self.batch_size):
            sample = []
            sample_similar = []
            for i in range(self.batch_size):
                sample_i = self.sample()
                if self.mode == "train" and self.similar_cnt > 0:
                    sample.append(sample_i[0])
                    sample_similar.append(sample_i[1])
                else:
                    sample.append(sample_i)
            sample = np.concatenate(sample, axis=0)
            if self.mode == "train" and self.similar_cnt > 0:
                sample_similar = np.concatenate(sample_similar, axis=0)
                yield sample, sample_similar  # [b, 48, 125], [b, 10, 48, 125]
            else:
                yield sample

    def onebyone_generator(self):
        all_subs = list(self.data.keys())
        if self.is_debug:
            all_subs = [all_subs[0]]  # debug 模式样本数 90

        for sub in all_subs:
            data_s = self.data[sub]
            if self.similar_cnt > 0:
                similar_pool = self.data_simar_pool[sub]

            all_acts = list(data_s.keys())
            if self.is_debug:
                all_acts = [all_acts[0]]
            for act in all_acts:
                seq = data_s[act]
                seq_len = seq.shape[0]

                for i in range(0, seq_len - self.t_total, self.t_his):  # step 取 25
                    data = seq[None, i: i + self.t_total]  # 1, 125, 17, 3
                    data = data[:, :, 1:, :].reshape(1, self.t_total, -1).transpose(0, 2, 1)  # 1, 48, 125
                    yield data

    def get_test_similat_gt_like_dlow(self):
        # todo 这部分在测试时找相似伪真值的结果并没有去做 limb 统一
        all_data = self.get_all_test_sample_inorder() # 188, 42, 75

        all_start_pose = all_data[:, :, self.t_his - 1]  # n, 48
        pd = squareform(pdist(all_start_pose))
        similar_gts = []
        num_mult = []
        for i in range(pd.shape[0]):
            ind = np.nonzero(np.logical_and(pd[i] < self.multimodal_threshold, pd[i] > 0))
            choosed_pseudo = all_data[ind][:, :, self.t_his:]  # n, 48, 100
            # todo 这里加入统一 limb 的操作
            if choosed_pseudo.shape[0] > 0:
                # 先转化为相对向量
                normalized_vector = choosed_pseudo.reshape(choosed_pseudo.shape[0], -1, 3, self.t_pred)  # n, 16, 3, 100
                normalized_vector = np.concatenate([np.zeros_like(normalized_vector[:, 0:1, :, :]), normalized_vector],
                                                   axis=1)  # n, 17, 3, 100
                normalized_vector = normalized_vector.transpose(0, 3, 1, 2).reshape(
                    choosed_pseudo.shape[0] * self.t_pred, -1, 3)  # n*100, 17, 3
                normalized_vector = self.coordinate_to_normalized_vector(normalized_vector)
                normalized_vector = normalized_vector.reshape(choosed_pseudo.shape[0], self.t_pred, -1, 3)
                # 将相对向量重整为 统一 limb

                x0 = np.copy(all_data[i, :, self.t_his]).reshape(1, 1, -1, 3)  # 1, 1, 16, 3
                x0 = np.concatenate([np.zeros_like(x0[:, :, 0:1, :]), x0], axis=2)  # 1, 1, 17, 3
                choosed_pseudo = self.normalized_vector_to_adpative_coordinate(normalized_vector,
                                                                               x0=x0)  # [n, 100, 17, 3]
                choosed_pseudo = choosed_pseudo[:, :, 1:, :].reshape(choosed_pseudo.shape[0], self.t_pred,
                                                                     -1).transpose(0, 2, 1)

            similar_gts.append(choosed_pseudo)  # n/0, 48, 100
            num_mult.append(len(ind[0]))

        num_mult = np.array(num_mult)

        print(f'#0 future: {len(np.where(num_mult == 0)[0])}/{pd.shape[0]}')  # 73/5168
        print(f'#<=9 future: {len(np.where(num_mult <= 8)[0])}/{pd.shape[0]}')  # 1139
        print(f'#>9 future: {len(np.where(num_mult > 8)[0])}/{pd.shape[0]}')  # 4029
        # todo: test 空数组的 [351, 352, 353, 356, 373, 374, 553, 579, 877, 878, 879, 886, 899, 903, 904, 935, 988, 990, 1009, 1158, 1201, 1336, 1480, 1603, 1849, 1932, 1935, 1983, 1984, 1990, 1998, 2058, 2215, 2216, 2223, 2371, 2374, 2384, 2614, 2615, 2697, 3494, 3504, 3505, 3506, 3512, 3514, 3515, 3516, 3517, 3531, 3570, 3574, 3575, 3581, 3585, 3653, 3885, 3908, 3970, 4323, 4341, 4417, 4568, 4623, 4854, 4855, 4856, 4857, 4858, 4874, 4898, 4901]
        self.similat_gt_like_dlow = similar_gts  # # n/0, 48, 100

    def get_all_test_sample_inorder(self):
        all_data = []
        all_subs = list(self.data.keys())
        for sub in all_subs:
            data_s = self.data[sub]
            all_acts = list(data_s.keys())
            for act in all_acts:
                seq = data_s[act]
                seq_len = seq.shape[0]

                for i in range(0, seq_len - self.t_total, self.t_his):  # step 取 25
                    data = seq[None, i: i + self.t_total]  # 1, 125, 17, 3
                    data = data[:, :, 1:, :].reshape(1, self.t_total, -1).transpose(0, 2, 1)  # 1, 48, 125
                    all_data.append(data)

        all_data = np.concatenate(all_data, axis=0)  # n, 48, 125
        return all_data

    def normalized_vector_to_adpative_coordinate(self, similar_pool, x0=None):
        """
        x: [bs,..., jn, 3] or [bs,..., jn-1, 3] if invert [18627, 125, 16, 3]
        x0: [1,..., jn, 3] 序列第一帧，其中第一个关节点变为 0 [1, 1, 17, 3]
        parents: [-1,0,1 ...]
        """

        jn = x0.shape[-2]
        limb_l = np.linalg.norm(x0[..., 1:, :] - x0[..., self.parents_17[1:], :], axis=-1, keepdims=True)  # 1, 1, 16, 1
        xt = similar_pool * limb_l
        hip = np.zeros_like(xt[..., :1, :])  # 18627, 125, 1, 3
        xt = np.concatenate([hip, xt], axis=-2)  # # 18627, 125, 17, 3
        for i in range(1, jn):
            xt[..., i, :] = xt[..., self.parents_17[i], :] + xt[..., i, :]
        return xt

    def coordinate_to_normalized_vector(self, x):
        """
        x: n, 17, 3
        parents_17:
        """
        xt = x[..., 1:, :] - x[..., self.parents_17[1:], :]
        xt = xt / np.linalg.norm(xt, axis=-1, keepdims=True)  # 标准化，成为单位向量 todo: parents_17 是否适应 16 个关节点的情况
        return xt  # n, 16, 3

    def coordinate_to_normalized_vector_torch(self, x):
        """
        x: n, 17, 3
        parents_17:
        """
        xt = x[..., 1:, :] - x[..., self.parents_17[1:], :]
        xt = xt / torch.norm(xt, dim=-1, keepdim=True)  # 标准化，成为单位向量 todo: parents_17 是否适应 16 个关节点的情况
        return xt  # n, 16, 3


class MaoweiGSPS_Dynamic_Seq_Humaneva_ExpandDataset_T1():
    def __init__(self, t_his=15, t_pred=60, dynamic_sub_len=2000, batch_size=8,
                 data_path=r"./dataset",
                 similar_idx_path=r"./dataset/data_multi_modal/t_his25_1_thre0.500_t_pred100_thre0.100_filtered_dlow.npz",
                 similar_pool_path=r"./dataset/data_multi_modal/data_candi_t_his25_t_pred100_skiprate20.npz",
                 subjects={"train": [f"S{i}" for i in [1, 5, 6, 7, 8]], "test": [f"S{i}" for i in [9, 11]]},
                 joint_used_17=[0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27],
                 parents_17=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15], mode="train",
                 multimodal_threshold=0.5, is_debug=False):

        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.dynamic_sub_len = dynamic_sub_len
        self.batch_size = batch_size
        self.subjects = subjects[mode]
        self.mode = mode
        self.multimodal_threshold = multimodal_threshold
        self.is_debug = is_debug

        self.actions = "all"

        self.parents_17 = parents_17

        data_o = np.load(os.path.join(data_path, "data_3d_humaneva15.npz"), allow_pickle=True)['positions_3d'].item()
        self.data = dict(
            filter(lambda x: x[0] in self.subjects, data_o.items()))  # dict [n, 125, 17, 3], hip 不为 0，但其他点会对hip相对化

        if self.mode == 'train':
            self.data['Train/S3'].pop('Walking 1 chunk0')
            self.data['Train/S3'].pop('Walking 1 chunk2')
        else:
            self.data['Validate/S3'].pop('Walking 1 chunk4')

        if mode == "train":
            self.data_similar_idx = np.load(similar_idx_path, allow_pickle=True)['data_multimodal'].item()  # dict

            similar_pool = np.load(similar_pool_path, allow_pickle=True)[
                'data_candidate.npy']  # [18627, 125, 16, 3], 其实是从整个数据集上面抽取出来的 18627 个样本，他们包含 16 个点的运动序列，用单位向量表示，消除个体差异
            self.data_simar_pool = {}  # dict [18627, 125, 17, 3] hip 统统为 0

        for key in list(self.data.keys()):
            self.data[key] = dict(filter(lambda x: (self.actions == 'all' or
                                                 all([a in x[0] for a in self.actions]))
                                                and x[1].shape[0] >= self.t_total, self.data[key].items()))
            if len(self.data[key]) == 0:
                self.data.pop(key)

        for sub in self.data.keys():
            for action in self.data[sub].keys():
                seq = self.data[sub][action][:, joint_used_17, :]  # n, 17, 3
                seq[:, 1:] -= seq[:, :1]  # 相对化
                self.data[sub][action] = seq

                if mode == "train" and (sub not in self.data_simar_pool.keys()):
                    x0 = np.copy(seq[None, :1, ...])  # 1, 1, 17, 3 序列第一帧
                    x0[:, :, 0] = 0  # 第一帧的第一个关节点
                    self.data_simar_pool[sub] = self.normalized_vector_to_adpative_coordinate(similar_pool, x0=x0)

        print(
            f"{mode} Data Loaded, dynamic_sub_len: {dynamic_sub_len}, batch_size: {batch_size}!")

    def sample(self):
        while True:
            subject = np.random.choice(self.subjects)
            dict_s = self.data[subject]
            action = np.random.choice(list(dict_s.keys()))
            seq = dict_s[action]
            if seq.shape[0] > self.t_total:
                break

        frame_start = np.random.randint(seq.shape[0] - self.t_total)
        frame_end = frame_start + self.t_total
        data = seq[frame_start: frame_end]  # 125, 17, 3

        pool = self.data_simar_pool[subject]
        idx_multi = self.data_similar_idx[subject][action][frame_start]
        data_similar = pool[idx_multi]  # [n, 125, 17, 3]

        if len(data_similar) > 0:
            data_similar[:, :self.t_his] = data[None, ...][:, :self.t_his]  # 用原来的拼接

        # todo: 以 1:3 的概率从 GT 和 pseudo 中采样
        if len(data_similar) == 0:
            choose_expand_data = data
        else:
            ran = np.random.uniform()
            if ran < 0.3333 or len(data_similar) == 0:
                choose_expand_data = data
            else:
                choose_idx_of_expand_data = np.random.randint(len(data_similar))
                choose_expand_data = data_similar[choose_idx_of_expand_data]

        choose_expand_data = choose_expand_data[:, 1:, :].reshape(1, self.t_total, -1).transpose(0, 2, 1)  # 1, 48, 125
        return choose_expand_data


    def batch_generator(self):
        if self.is_debug:
            self.dynamic_sub_len = 200

        for i in range(self.dynamic_sub_len // self.batch_size):
            sample = []
            for i in range(self.batch_size):
                sample_i = self.sample()
                sample.append(sample_i)
            sample = np.concatenate(sample, axis=0)
            yield sample

    def onebyone_generator_for_test(self):
        all_subs = list(self.data.keys())
        if self.is_debug:
            all_subs = [all_subs[0]]  # debug 模式样本数 90

        for sub in all_subs:
            data_s = self.data[sub]
            all_acts = list(data_s.keys())
            if self.is_debug:
                all_acts = [all_acts[0]]
            for act in all_acts:
                seq = data_s[act]
                seq_len = seq.shape[0]

                for i in range(0, seq_len - self.t_total, self.t_his):  # step 取 25
                    data = seq[None, i: i + self.t_total]  # 1, 125, 17, 3

                    data = data[:, :, 1:, :].reshape(1, self.t_total, -1).transpose(0, 2, 1)  # 1, 48, 125
                    yield data

    def get_test_similat_gt_like_dlow(self):
        # todo 这部分在测试时找相似伪真值的结果并没有去做 limb 统一
        all_data = self.get_all_test_sample_inorder() # 188, 42, 75

        all_start_pose = all_data[:, :, self.t_his - 1]  # n, 48
        pd = squareform(pdist(all_start_pose))
        similar_gts = []
        num_mult = []
        for i in range(pd.shape[0]):
            ind = np.nonzero(np.logical_and(pd[i] < self.multimodal_threshold, pd[i] > 0))
            choosed_pseudo = all_data[ind][:, :, self.t_his:]  # n, 48, 100
            # todo 这里加入统一 limb 的操作
            if choosed_pseudo.shape[0] > 0:
                # 先转化为相对向量
                normalized_vector = choosed_pseudo.reshape(choosed_pseudo.shape[0], -1, 3, self.t_pred)  # n, 16, 3, 100
                normalized_vector = np.concatenate([np.zeros_like(normalized_vector[:, 0:1, :, :]), normalized_vector],
                                                   axis=1)  # n, 17, 3, 100
                normalized_vector = normalized_vector.transpose(0, 3, 1, 2).reshape(
                    choosed_pseudo.shape[0] * self.t_pred, -1, 3)  # n*100, 17, 3
                normalized_vector = self.coordinate_to_normalized_vector(normalized_vector)
                normalized_vector = normalized_vector.reshape(choosed_pseudo.shape[0], self.t_pred, -1, 3)
                # 将相对向量重整为 统一 limb

                x0 = np.copy(all_data[i, :, self.t_his]).reshape(1, 1, -1, 3)  # 1, 1, 16, 3
                x0 = np.concatenate([np.zeros_like(x0[:, :, 0:1, :]), x0], axis=2)  # 1, 1, 17, 3
                choosed_pseudo = self.normalized_vector_to_adpative_coordinate(normalized_vector,
                                                                               x0=x0)  # [n, 100, 17, 3]
                choosed_pseudo = choosed_pseudo[:, :, 1:, :].reshape(choosed_pseudo.shape[0], self.t_pred,
                                                                     -1).transpose(0, 2, 1)

            similar_gts.append(choosed_pseudo)  # n/0, 48, 100
            num_mult.append(len(ind[0]))

        num_mult = np.array(num_mult)

        print(f'#0 future: {len(np.where(num_mult == 0)[0])}/{pd.shape[0]}')  # 73/5168
        print(f'#<=9 future: {len(np.where(num_mult <= 8)[0])}/{pd.shape[0]}')  # 1139
        print(f'#>9 future: {len(np.where(num_mult > 8)[0])}/{pd.shape[0]}')  # 4029
        # todo: test 空数组的 [351, 352, 353, 356, 373, 374, 553, 579, 877, 878, 879, 886, 899, 903, 904, 935, 988, 990, 1009, 1158, 1201, 1336, 1480, 1603, 1849, 1932, 1935, 1983, 1984, 1990, 1998, 2058, 2215, 2216, 2223, 2371, 2374, 2384, 2614, 2615, 2697, 3494, 3504, 3505, 3506, 3512, 3514, 3515, 3516, 3517, 3531, 3570, 3574, 3575, 3581, 3585, 3653, 3885, 3908, 3970, 4323, 4341, 4417, 4568, 4623, 4854, 4855, 4856, 4857, 4858, 4874, 4898, 4901]
        self.similat_gt_like_dlow = similar_gts  # # n/0, 48, 100

    def get_all_test_sample_inorder(self):
        all_data = []
        all_subs = list(self.data.keys())
        for sub in all_subs:
            data_s = self.data[sub]
            all_acts = list(data_s.keys())
            for act in all_acts:
                seq = data_s[act]
                seq_len = seq.shape[0]

                for i in range(0, seq_len - self.t_total, self.t_his):  # step 取 25
                    data = seq[None, i: i + self.t_total]  # 1, 125, 17, 3
                    data = data[:, :, 1:, :].reshape(1, self.t_total, -1).transpose(0, 2, 1)  # 1, 48, 125
                    all_data.append(data)

        all_data = np.concatenate(all_data, axis=0)  # n, 48, 125
        return all_data

    def normalized_vector_to_adpative_coordinate(self, similar_pool, x0=None):
        """
        x: [bs,..., jn, 3] or [bs,..., jn-1, 3] if invert [18627, 125, 16, 3]
        x0: [1,..., jn, 3] 序列第一帧，其中第一个关节点变为 0 [1, 1, 17, 3]
        parents: [-1,0,1 ...]
        """

        jn = x0.shape[-2]
        limb_l = np.linalg.norm(x0[..., 1:, :] - x0[..., self.parents_17[1:], :], axis=-1, keepdims=True)  # 1, 1, 16, 1
        xt = similar_pool * limb_l
        hip = np.zeros_like(xt[..., :1, :])  # 18627, 125, 1, 3
        xt = np.concatenate([hip, xt], axis=-2)  # # 18627, 125, 17, 3
        for i in range(1, jn):
            xt[..., i, :] = xt[..., self.parents_17[i], :] + xt[..., i, :]
        return xt

    def coordinate_to_normalized_vector(self, x):
        """
        x: n, 17, 3
        parents_17:
        """
        xt = x[..., 1:, :] - x[..., self.parents_17[1:], :]
        xt = xt / np.linalg.norm(xt, axis=-1, keepdims=True)  # 标准化，成为单位向量 todo: parents_17 是否适应 16 个关节点的情况
        return xt  # n, 16, 3

    def coordinate_to_normalized_vector_torch(self, x):
        """
        x: n, 17, 3
        parents_17:
        """
        xt = x[..., 1:, :] - x[..., self.parents_17[1:], :]
        xt = xt / torch.norm(xt, dim=-1, keepdim=True)  # 标准化，成为单位向量 todo: parents_17 是否适应 16 个关节点的情况
        return xt  # n, 16, 3



if __name__ == '__main__':
        I17_plot = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
        J17_plot = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        LR17_plot = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])

        from datas.draw_pictures import draw_multi_seqs_2d

        # train_d = MaoweiGSPS_Dynamic(mode="train", is_debug=True)
        # train_gen = train_d.batch_generator()
        # for i, sample in enumerate(train_gen):
        #     print(f"{i} : {sample[0].shape[0]}")
        #     idx = np.random.choice(np.arange(8))
        #     gt = sample[0]  # 8, 48, 125
        #     sims = sample[1]  # 8, 10, 48, 125
        #     all_to_draw = np.concatenate((gt[idx][None, ...], sims[idx]), axis=0).reshape(
        #         (1 + train_d.similar_cnt, -1, 3, train_d.t_total))[:, :, [0, 2], :]  # 1+10, 16, 2, 125
        #     all_to_draw = np.concatenate(
        #         (np.expand_dims(np.mean(all_to_draw[:, [0, 3], :, :], axis=1), axis=1), all_to_draw), axis=1)
        #     all_to_draw *= 1000
        #
        #     draw_acc = [acc for acc in range(0, 125, 5)]
        #     all_to_draw = all_to_draw[:, :, :, draw_acc]
        #
        #     # draw_multi_seqs_2d(all_to_draw, gt_cnt=1, t_his=5, I=I17_plot, J=J17_plot, LR=LR17_plot, full_path=f"{i}-{idx}_train_p17.png")
        #     pass

        test_d = MaoweiGSPS_Dynamic_Seq_Humaneva(mode="test", similar_cnt=0, is_debug=True)
        test_d.get_test_similat_gt_like_dlow()
        test_gen = test_d.onebyone_generator()
        for i, sample in enumerate(test_gen):  # 1, 125, 17, 3
            similars = test_d.similat_gt_like_dlow[i]  # n/0, 48, 100
            print(f"{i} : {similars.shape[0]}")
            pass
        pass
