#!/usr/bin/env python
# encoding: utf-8
'''
@project : m3day32022
@file    : h36m_dataset_classifier.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-06-08 16:14
'''
import numpy as np
import os
from scipy.spatial.distance import pdist, squareform
import torch



class MaoweiGSPS_Dynamic_Seq_Classifier_H36m():
    def __init__(self, t_his=25, t_pred=100, similar_cnt=10, dynamic_sub_len=5000, batch_size=8,
                 data_path=r"F:\model_report_data\mocap_motion_prediction\data\h36m_dlow_origin",
                 similar_idx_path=r"F:\model_report_data\stochastic_prediction\gsps\data_multi_modal\t_his25_1_thre0.500_t_pred100_thre0.100_filtered_dlow.npz",
                 similar_pool_path=r"F:\model_report_data\stochastic_prediction\gsps\data_multi_modal\data_candi_t_his25_t_pred100_skiprate20.npz",
                 subjects={"train": [f"S{i}" for i in [1, 5, 6, 7, 8]], "test": [f"S{i}" for i in [9, 11]]},
                 joint_used_17=[0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27],
                 parents_17=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15], mode="train",
                 multimodal_threshold=0.5, is_debug=False):

        self.actions_15 = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        self.actionos_5 = [['Directions', 'Discussion', 'Greeting', 'Photo', 'Posing', 'Purchases', 'WalkDog', 'Waiting'],
                           ['Eating', 'Phoning', 'Sitting', 'Smoking'],
                           ['SittingDown'],
                           ['Walking'],
                           ['WalkTogether']]

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

        self.parents_17 = parents_17

        data_o = np.load(os.path.join(data_path, "data_3d_h36m.npz"), allow_pickle=True)['positions_3d'].item()
        self.data = dict(
            filter(lambda x: x[0] in self.subjects, data_o.items()))  # dict [n, 125, 17, 3], hip 不为 0，但其他点会对hip相对化

        if mode == "train" and similar_cnt > 0:
            self.data_similar_idx = np.load(similar_idx_path, allow_pickle=True)['data_multimodal'].item()  # dict

            similar_pool = np.load(similar_pool_path, allow_pickle=True)[
                'data_candidate.npy']  # [18627, 125, 16, 3], 其实是从整个数据集上面抽取出来的 18627 个样本，他们包含 16 个点的运动序列，用单位向量表示，消除个体差异
            self.data_simar_pool = {}  # dict [18627, 125, 17, 3] hip 统统为 0

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
        subject = np.random.choice(self.subjects)
        dict_s = self.data[subject]
        action = np.random.choice(list(dict_s.keys()))

        action_idx_15 = self.actions_15.index(action.split(" ")[0])
        for itemidx, act_set in enumerate(self.actionos_5):
            if self.actions_15[action_idx_15] in act_set:
                action_idx_5 = itemidx
                break

        seq = dict_s[action]

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
                [data_similar, np.zeros_like(data[None, ...][[0] * (self.similar_cnt - data_similar.shape[0])])],
                axis=0)  # repeat

            data = data[:, 1:, :].reshape(1, self.t_total, -1).transpose(0, 2, 1)  # 1, 48, 125
            data_similar = data_similar[:, :, 1:, :].reshape(1, self.similar_cnt, self.t_total, -1).transpose(0, 1, 3,
                                                                                                              2)  # 1, 10, 48, 125
            return data, action_idx_15, action_idx_5, data_similar
        else:
            data = data[:, 1:, :].reshape(1, self.t_total, -1).transpose(0, 2, 1)  # 1, 48, 125
            return data, action_idx_15, action_idx_5

    def batch_generator(self):
        if self.is_debug:
            self.dynamic_sub_len = 200

        for i in range(self.dynamic_sub_len // self.batch_size):
            sample = []
            sample_similar = []
            act_idx_15 = []
            act_idx_5 = []
            for i in range(self.batch_size):
                sample_i = self.sample()
                if self.mode == "train" and self.similar_cnt > 0:
                    sample.append(sample_i[0])
                    act_idx_15.append(sample_i[1])
                    act_idx_5.append(sample_i[2])
                    sample_similar.append(sample_i[3])
                else:
                    sample.append(sample_i[0])
                    act_idx_15.append(sample_i[1])
                    act_idx_5.append(sample_i[2])
            sample = np.concatenate(sample, axis=0)
            act_idx_15 = np.array(act_idx_15).reshape(-1, 1)
            act_idx_5 = np.array(act_idx_5).reshape(-1, 1)

            if self.mode == "train" and self.similar_cnt > 0:
                sample_similar = np.concatenate(sample_similar, axis=0)
                yield sample, act_idx_15, act_idx_5, sample_similar  # [b, 48, 125], [b, 10, 48, 125]
            else:
                yield sample, act_idx_15, act_idx_5

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
                action_idx_15 = self.actions_15.index(act.split(" ")[0])
                for itemidx, act_set in enumerate(self.actionos_5):
                    if self.actions_15[action_idx_15] in act_set:
                        action_idx_5 = itemidx
                        break

                seq = data_s[act]
                seq_len = seq.shape[0]

                for i in range(0, seq_len - self.t_total, self.t_his):  # step 取 25
                    data = seq[None, i: i + self.t_total]  # 1, 125, 17, 3
                    data = data[:, :, 1:, :].reshape(1, self.t_total, -1).transpose(0, 2, 1)  # 1, 48, 125

                    action_idx_15 = np.array(action_idx_15).reshape(-1, 1)
                    action_idx_5 = np.array(action_idx_5).reshape(-1, 1)
                    yield data, action_idx_15, action_idx_5


    def get_test_similat_gt_like_dlow(self):
        # todo 这部分在测试时找相似伪真值的结果并没有去做 limb 统一
        all_data = self.get_all_test_sample_inorder() # []

        all_start_pose = all_data[:, :, self.t_his - 1]  # n, 48
        pd = squareform(pdist(all_start_pose))
        similar_gts = []
        num_mult = []
        for i in range(pd.shape[0]):
            ind = np.nonzero(np.logical_and(pd[i] < self.multimodal_threshold, pd[i] > 0))
            choosed_pseudo = all_data[ind][:, :, self.t_his:] # n, 48, 100
            # todo 这里加入统一 limb 的操作
            if choosed_pseudo.shape[0]> 0:
                # 先转化为相对向量
                normalized_vector = choosed_pseudo.reshape(choosed_pseudo.shape[0], -1, 3, self.t_pred)  # n, 16, 3, 100
                normalized_vector = np.concatenate([np.zeros_like(normalized_vector[:, 0:1, :, :]), normalized_vector], axis=1) # n, 17, 3, 100
                normalized_vector = normalized_vector.transpose(0, 3, 1, 2).reshape(choosed_pseudo.shape[0] * self.t_pred, -1, 3) # n*100, 17, 3
                normalized_vector = self.coordinate_to_normalized_vector(normalized_vector)
                normalized_vector = normalized_vector.reshape(choosed_pseudo.shape[0], self.t_pred, -1, 3)
                # 将相对向量重整为 统一 limb

                x0 = np.copy(all_data[i, :, self.t_his]).reshape(1, 1, -1, 3)  # 1, 1, 16, 3
                x0 = np.concatenate([np.zeros_like(x0[:, :, 0:1, :]), x0], axis=2) # 1, 1, 17, 3
                choosed_pseudo = self.normalized_vector_to_adpative_coordinate(normalized_vector, x0=x0) # [n, 100, 17, 3]
                choosed_pseudo = choosed_pseudo[:, :, 1:, :].reshape(choosed_pseudo.shape[0], self.t_pred, -1).transpose(0, 2, 1)

            similar_gts.append(choosed_pseudo)  # n/0, 48, 100
            num_mult.append(len(ind[0]))

        num_mult = np.array(num_mult)

        print(f'#0 future: {len(np.where(num_mult == 0)[0])}/{pd.shape[0]}') # 73/5168
        print(f'#<=9 future: {len(np.where(num_mult <= 9)[0])}/{pd.shape[0]}') # 1139
        print(f'#>9 future: {len(np.where(num_mult > 9)[0])}/{pd.shape[0]}') # 4029
        # todo: test 空数组的 [351, 352, 353, 356, 373, 374, 553, 579, 877, 878, 879, 886, 899, 903, 904, 935, 988, 990, 1009, 1158, 1201, 1336, 1480, 1603, 1849, 1932, 1935, 1983, 1984, 1990, 1998, 2058, 2215, 2216, 2223, 2371, 2374, 2384, 2614, 2615, 2697, 3494, 3504, 3505, 3506, 3512, 3514, 3515, 3516, 3517, 3531, 3570, 3574, 3575, 3581, 3585, 3653, 3885, 3908, 3970, 4323, 4341, 4417, 4568, 4623, 4854, 4855, 4856, 4857, 4858, 4874, 4898, 4901]
        self.similat_gt_like_dlow = similar_gts # # n/0, 48, 100

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
        x: [bs,..., jn, 3] or [bs,..., jn-1, 3] if invert [n, 125, 16, 3]
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
        xt = xt / np.linalg.norm(xt, axis=-1, keepdims=True) # 标准化，成为单位向量 todo: parents_17 是否适应 16 个关节点的情况
        return xt # n, 16, 3

    def coordinate_to_normalized_vector_torch(self, x):
        """
        x: n, 17, 3
        parents_17:
        """
        xt = x[..., 1:, :] - x[..., self.parents_17[1:], :]
        xt = xt / torch.norm(xt, dim=-1, keepdim=True) # 标准化，成为单位向量 todo: parents_17 是否适应 16 个关节点的情况
        return xt # n, 16, 3


if __name__ == '__main__':
    ds = MaoweiGSPS_Dynamic_Seq_Classifier_H36m(data_path=r"G:\second_model_report_data\writing_paper2022\rebuttal\codes\other_evaluation_metrics\gsps_dataset")