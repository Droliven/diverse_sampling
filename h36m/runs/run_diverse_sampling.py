#!/usr/bin/env python
# encoding: utf-8
'''
@project : baseresample_likegsps
@file    : run_decoupled.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-13 10:57
'''

from ..datas import MaoweiGSPS_Dynamic_Seq_H36m, draw_multi_seqs_2d, get_dct_matrix, dct_transform_torch, reverse_dct_torch
from ..nets import DiverseSampling
from ..nets import CVAE
from ..configs import ConfigDiverseSampling
from .losses import loss_kl_normal, loss_recons_adelike, loss_recons_mmadelike, loss_diversity_hinge_divide, \
    loss_diversity_hinge_between_two_part, \
    loss_limb_length_t2, loss_recover_history_t2, loss_valid_angle_t2, compute_diversity, \
    compute_diversity_between_twopart, compute_ade, compute_fde, compute_mmade, compute_mmfde, \
    compute_bone_percent_error, compute_angle_error

from torch.optim import Adam, lr_scheduler
import torch
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pprint import pprint
import random
import numpy as np
import json
import pickle


class RunDiverseSampling():
    def __init__(self, exp_name="", device="cuda:0", num_works=0, is_debug=False, args=None):
        super(RunDiverseSampling, self).__init__()

        self.is_debug = is_debug

        # 参数
        self.start_epoch = 1
        self.best_accuracy = 1e15

        self.cfg = ConfigDiverseSampling(exp_name=exp_name, device=device, num_works=num_works)

        print("\n================== Arguments =================")
        pprint(vars(args), indent=4)
        print("==========================================\n")

        print("\n================== Configs =================")
        pprint(vars(self.cfg), indent=4)
        print("==========================================\n")

        save_dict = {"args": args.__dict__, "cfgs": self.cfg.__dict__}
        save_json = json.dumps(save_dict)

        with open(os.path.join(self.cfg.ckpt_dir, "config.json"), 'w', encoding='utf-8') as f:
            f.write(save_json)

        # 模型
        self.model_t1 = CVAE(node_n=self.cfg.node_n, hidden_dim=self.cfg.hidden_dim, z_dim=self.cfg.z_dim,
                                     dct_n=self.cfg.dct_n, dropout_rate=self.cfg.dropout_rate)
        self.model = DiverseSampling(node_n=self.cfg.node_n, hidden_dim=self.cfg.hidden_dim,
                                                      base_dim=self.cfg.base_dim, base_num_p1=self.cfg.base_num_p1, base_num_p2=self.cfg.base_num_p2,
                                                      z_dim=self.cfg.z_dim, dct_n=self.cfg.dct_n,
                                                      dropout_rate=self.cfg.dropout_rate)

        if self.cfg.device != "cpu":
            self.model_t1.cuda(self.cfg.device)
            self.model.cuda(self.cfg.device)

        print(">>> total params of {}: {:.6f}M\n".format("t1", sum(
            p.numel() for p in self.model_t1.parameters()) / 1000000.0))
        print(">>> total params of {}: {:.6f}M\n".format(exp_name,
                                                         sum(p.numel() for p in self.model.parameters()) / 1000000.0))

        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.lr_t2)

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - self.cfg.epoch_fix_t2) / float(self.cfg.epoch_t2 - self.cfg.epoch_fix_t2 + 1)
            return lr_l

        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)

        # 导入参数并冻结
        model_t1_state = torch.load(self.cfg.model_path_t1, map_location=self.cfg.device)
        self.model_t1.load_state_dict(model_t1_state["model"])
        print("{} loaded from {}".format("model_t1", self.cfg.model_path_t1))
        for p in self.model_t1.parameters():
            p.requires_grad = False
        self.model_t1.eval()

        # 数据
        self.train_data = MaoweiGSPS_Dynamic_Seq_H36m(data_path=self.cfg.base_data_dir,
                                                 similar_idx_path=self.cfg.similar_idx_path,
                                                 similar_pool_path=self.cfg.similar_pool_path, t_his=self.cfg.t_his,
                                                 t_pred=self.cfg.t_pred, similar_cnt=self.cfg.train_similar_cnt,
                                                 dynamic_sub_len=self.cfg.sub_len_train,
                                                 batch_size=self.cfg.train_batch_size,
                                                 joint_used_17=self.cfg.joint_used, subjects=self.cfg.subjects,
                                                 parents_17=self.cfg.parents,
                                                 mode="train", multimodal_threshold=self.cfg.multimodal_threshold,
                                                 is_debug=self.is_debug)
        self.test_data = MaoweiGSPS_Dynamic_Seq_H36m(data_path=self.cfg.base_data_dir,
                                                similar_idx_path=self.cfg.similar_idx_path,
                                                similar_pool_path=self.cfg.similar_pool_path, t_his=self.cfg.t_his,
                                                t_pred=self.cfg.t_pred, similar_cnt=0,
                                                dynamic_sub_len=self.cfg.sub_len_train,
                                                batch_size=self.cfg.test_batch_size,
                                                joint_used_17=self.cfg.joint_used, subjects=self.cfg.subjects,
                                                parents_17=self.cfg.parents,
                                                mode="test", multimodal_threshold=self.cfg.multimodal_threshold,
                                                is_debug=self.is_debug)
        self.test_data.get_test_similat_gt_like_dlow()

        self.valid_angle = pickle.load(open(self.cfg.valid_angle_path, "rb"))  # dict 13
        print(f"{'valid angle'} loaded from {self.cfg.valid_angle_path} !")

        ## dct
        self.dct_m, self.i_dct_m = get_dct_matrix(self.cfg.t_total)
        if self.cfg.device != "cpu":
            self.dct_m = torch.from_numpy(self.dct_m).float().cuda()
            self.i_dct_m = torch.from_numpy(self.i_dct_m).float().cuda()

        self.summary = SummaryWriter(self.cfg.ckpt_dir)

    def _sample_weight_gumbel_softmax(self, logits, temperature=1, eps=1e-20):
        # b*h, 1, 10
        assert temperature > 0, "temperature must be greater than 0 !"

        U = torch.rand(logits.shape, device=logits.device)
        g = -torch.log(-torch.log(U + eps) + eps)

        y = logits + g
        y = y / temperature
        y = torch.softmax(y, dim=-1)
        return y

    def save(self, checkpoint_path, epoch, curr_err):
        state = {
            "epoch": epoch,
            "lr": self.scheduler.get_last_lr()[0],
            "curr_err": curr_err,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=self.cfg.device)
        self.model.load_state_dict(state["model"])
        # self.optimizer.load_state_dict(state["optimizer"])
        # self.lr = state["lr"]
        # self.start_epoch = state["epoch"] + 1
        curr_err = state["curr_err"]
        print(
            "load from epoch {}, lr {}, curr_avg {}.".format(state["epoch"], self.scheduler.get_last_lr()[0], curr_err))

    def train(self, epoch, draw=False):
        self.model.train()

        average_allloss = 0
        average_kls_p1 = 0

        average_adeerrors = 0
        average_mmadeerrors = 0
        average_hinges = 0

        average_his = 0
        average_limblen = 0
        average_angle = 0

        dg = self.train_data.batch_generator()
        generator_len = self.cfg.sub_len_train // self.train_data.batch_size if not self.is_debug else 200 // self.train_data.batch_size
        draw_i = random.randint(0, generator_len - 1)
        for i, (datas, similars) in enumerate(dg):
            # [b, 48, 125], [b, 10, 48, 125]
            b, vc, t = datas.shape
            # skip the last batch if only have one sample for batch_norm layers
            if b == 1:
                continue

            self.global_step = (epoch - 1) * generator_len + i + 1

            datas = torch.from_numpy(datas).float().cuda(device=self.cfg.device)
            similars = torch.from_numpy(similars).float().cuda(device=self.cfg.device)
            eps = torch.randn((b, self.cfg.z_dim), device=self.cfg.device)
            # todo 训练过程要统一
            repeated_eps = torch.repeat_interleave(eps, repeats=self.cfg.nk, dim=0)

            with torch.no_grad():
                padded_inputs = datas[:, :, list(range(self.cfg.t_his)) + [self.cfg.t_his - 1] * self.cfg.t_pred]
                padded_inputs_dct = dct_transform_torch(padded_inputs, self.dct_m, dct_n=self.cfg.dct_n)  # b, 48, 10
                padded_inputs_dct = padded_inputs_dct.view(b, -1, 3 * self.cfg.dct_n)  # # b, 16, 3*10

            # >>> many bases
            logtics = torch.ones((b * self.cfg.nk, 1, self.cfg.base_num_p1), device=self.cfg.device) / self.cfg.base_num_p1  # b*h, 1, 10
            many_weights = self._sample_weight_gumbel_softmax(logtics, temperature=self.cfg.temperature_p1)  # b*h, 1, 10
            all_z_p1, all_mean_p1, all_logvar_p1 = self.model(condition=padded_inputs_dct, repeated_eps=repeated_eps, many_weights=many_weights,
                                                              temperature=None,
                                                              multi_modal_head=self.cfg.nk,
                                                              mode="p1")  # b*(10), 128

            all_z = all_z_p1

            all_outs_dct = self.model_t1.inference(
                condition=torch.repeat_interleave(padded_inputs_dct, repeats=self.cfg.nk, dim=0),
                z=all_z)  # b*h, 16, 30
            all_outs_dct = all_outs_dct.reshape(b * self.cfg.nk, -1, self.cfg.dct_n)  # b*h, 48, 10
            outputs = reverse_dct_torch(all_outs_dct, self.i_dct_m, self.cfg.t_total)  # b*h, 48, 125
            outputs = outputs.view(b, self.cfg.nk, -1, self.cfg.t_total)  # b, 50, 48, 125

            # loss
            kls_p1 = loss_kl_normal(all_mean_p1, all_logvar_p1)
            adeerrors = loss_recons_adelike(gt=datas[:, :, self.cfg.t_his:],
                                            pred=outputs[:, :, :, self.cfg.t_his:])
            mmadeerrors = loss_recons_mmadelike(similars=similars[:, :, :, self.cfg.t_his:],
                                                pred=outputs[:, :, :, self.cfg.t_his:])
            all_hinges = loss_diversity_hinge_divide(outputs[:, :, :, self.cfg.t_his:],
                                                     minthreshold=self.cfg.minthreshold,
                                                     seperate_head=self.cfg.seperate_head)

            recovrhis_err = 0  # loss_recover_history_t2(outputs[:, :, :, :self.cfg.t_his], datas[:, :, :self.cfg.t_his])  # 这里用 25 帧
            limblen_err = loss_limb_length_t2(pred=outputs, gt=datas, parent_17=self.cfg.parents)  # 这里用 125帧
            angle_err = 0 # loss_valid_angle_t2(outputs[:, :, :, self.cfg.t_his:], self.valid_angle, data="h36m")

            all_loss = kls_p1 * self.cfg.t2_kl_p1_weight \
                       + adeerrors * self.cfg.t2_ade_weight \
                       + mmadeerrors * self.cfg.t2_recons_mm_weight \
                       + all_hinges * self.cfg.t2_diversity_weight \
                       + limblen_err * self.cfg.t2_limblen_weight

            if angle_err > 0:
                all_loss += angle_err * self.cfg.t2_angle_weight

            self.optimizer.zero_grad()
            all_loss.backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), max_norm=100)
            self.optimizer.step()

            average_allloss += all_loss.cpu().data.numpy()
            average_kls_p1 += kls_p1.cpu().data.numpy()
            average_adeerrors += adeerrors.cpu().data.numpy()
            average_mmadeerrors += mmadeerrors.cpu().data.numpy()
            average_hinges += all_hinges.cpu().data.numpy()

            average_his += 0  # recovrhis_err.cpu().data.numpy()
            average_limblen += limblen_err.cpu().data.numpy()
            average_angle += 0  # angle_err.cpu().data.numpy()

            # 画图
            if draw:
                if i == draw_i:
                    bidx = 0
                    origin = datas[bidx:bidx + 1].detach().cpu().numpy()  # 1, 48, 125
                    origin = origin.reshape(1, -1, 3, self.cfg.t_total)  # 1, 16, 3, 125
                    origin = np.concatenate((np.expand_dims(np.mean(origin[:, [0, 3], :, :], axis=1), axis=1), origin),
                                            axis=1)  # # 1, 17, 3, 125
                    origin *= 1000

                    output = outputs[bidx, :self.cfg.seperate_head].reshape(self.cfg.seperate_head, -1, 3,
                                                                            self.cfg.t_total).detach().cpu().numpy()  # 50, 16, 3, 125
                    output = np.concatenate((np.expand_dims(np.mean(output[:, [0, 3], :, :], axis=1), axis=1), output),
                                            axis=1)  # # 50, 17, 3, 100
                    output *= 1000

                    all_to_draw = np.concatenate((origin, output), axis=0)
                    draw_acc = [acc for acc in range(0, all_to_draw.shape[-1], 5)]
                    all_to_draw = all_to_draw[:, :, :, draw_acc][:, :, [0, 2], :]

                    draw_multi_seqs_2d(all_to_draw, gt_cnt=1, t_his=5,
                                       I=self.cfg.I17_plot, J=self.cfg.J17_plot,
                                       LR=self.cfg.LR17_plot,
                                       full_path=os.path.join(self.cfg.ckpt_dir, "images",
                                                              f"train_epo{epoch}idx{draw_i}.png"))

        average_allloss /= (i + 1)
        average_kls_p1 /= (i + 1)
        average_adeerrors /= (i + 1)
        average_mmadeerrors /= (i + 1)
        average_hinges /= (i + 1)

        average_his /= (i + 1)
        average_limblen /= (i + 1)
        average_angle /= (i + 1)

        self.summary.add_scalar("loss/average_all", average_allloss, epoch)
        self.summary.add_scalar("loss/average_kls_p1", average_kls_p1, epoch)
        self.summary.add_scalar("loss/average_ades", average_adeerrors, epoch)
        self.summary.add_scalar("loss/average_mmades", average_mmadeerrors, epoch)
        self.summary.add_scalar("loss/average_hinges", average_hinges, epoch)

        self.summary.add_scalar(f"loss/averagerhis", average_his, epoch)
        self.summary.add_scalar(f"loss/averagelimblen", average_limblen, epoch)
        self.summary.add_scalar(f"loss/averageangle", average_angle, epoch)

        return average_allloss, average_adeerrors, average_mmadeerrors, average_hinges, average_kls_p1, average_limblen, average_angle


    def eval(self, epoch=-1, draw=False):
        self.model.eval()

        diversity = 0
        ade = 0
        fde = 0
        mmade = 0
        mmfde = 0
        bone = 0
        min_bone = 0
        max_bone = 0
        angle = 0
        # 画图 ------------------------------------------------------------------------------------------------------
        if not os.path.exists(os.path.join(self.cfg.ckpt_dir, "images", "sample")):
            os.makedirs(os.path.join(self.cfg.ckpt_dir, "images", "sample"))

        dg = self.test_data.onebyone_generator()
        generator_len = len(self.test_data.similat_gt_like_dlow) if not self.is_debug else 90

        draw_i = random.randint(0, generator_len - 1)

        for i, datas in enumerate(dg):
            # b, 48, 125
            b, vc, t = datas.shape
            similars = self.test_data.similat_gt_like_dlow[i]  # 0/n, 48, 100
            if similars.shape[0] == 0:  # todo 这会淡化误差
                continue

            datas = torch.from_numpy(datas).float().cuda(device=self.cfg.device)
            similars = torch.from_numpy(similars).float().cuda(device=self.cfg.device)

            with torch.no_grad():
                padded_inputs = datas[:, :, list(range(self.cfg.t_his)) + [self.cfg.t_his - 1] * self.cfg.t_pred]
                padded_inputs_dct = dct_transform_torch(padded_inputs, self.dct_m, dct_n=self.cfg.dct_n)  # b, 48, 10
                padded_inputs_dct = padded_inputs_dct.view(b, -1, 3 * self.cfg.dct_n)  # # b, 16, 3*10

                # eps = torch.randn((b, self.cfg.z_dim), device=self.cfg.device)
                # repeated_eps_1 = torch.repeat_interleave(eps, repeats=self.cfg.nk, dim=0)
                # repeated_eps_2 = torch.repeat_interleave(eps, repeats=self.cfg.nk, dim=0)
                # todo train generator eps 不共享
                repeated_eps_1 = torch.randn((b * self.cfg.nk, self.cfg.z_dim), device=self.cfg.device)

                logtics = torch.ones((b * self.cfg.nk, 1, self.cfg.base_num_p1),
                                     device=self.cfg.device) / self.cfg.base_num_p1  # b*h, 1, 10
                many_weights = self._sample_weight_gumbel_softmax(logtics, temperature=self.cfg.temperature_p1)  # b*h, 1, 10
                all_z_p1, all_mean_p1, all_logvar_p1 = self.model(condition=padded_inputs_dct,
                                                                  repeated_eps=repeated_eps_1, many_weights=many_weights,
                                                                  temperature=None,
                                                                  multi_modal_head=self.cfg.nk,
                                                                  mode="p1")  # b*(10), 128

                all_z = all_z_p1
                all_outs_dct = self.model_t1.inference(condition=torch.repeat_interleave(padded_inputs_dct, repeats=self.cfg.nk, dim=0), z=all_z)  # b*h, 16, 30
                all_outs_dct = all_outs_dct.reshape(b * self.cfg.nk, -1, self.cfg.dct_n)  # b*h, 48, 10
                outputs = reverse_dct_torch(all_outs_dct, self.i_dct_m, self.cfg.t_total)  # b*h, 48, 125
                outputs = outputs.view(self.cfg.nk, -1, self.cfg.t_total)[:, :, self.cfg.t_his:]  # 50, 48, 100

                # 两支一起平均
                cade = compute_ade(outputs, datas[:, :, self.cfg.t_his:])
                cfde = compute_fde(outputs, datas[:, :, self.cfg.t_his:])
                cmmade = compute_mmade(outputs, datas[:, :, self.cfg.t_his:], similars)
                cmmfde = compute_mmfde(outputs, datas[:, :, self.cfg.t_his:], similars)
                # cdiv = compute_diversity(pred=outputs).mean()
                cdiv = []
                for oidx in range(self.cfg.nk // self.cfg.seperate_head):
                    cdiv.append(compute_diversity(
                        outputs[oidx * self.cfg.seperate_head:(oidx + 1) * self.cfg.seperate_head, :,
                        :]))  # [10, 48, 100], [1, 48, 100]
                    for ojdx in range(oidx + 1, self.cfg.nk // self.cfg.seperate_head):
                        cdiv.append(compute_diversity_between_twopart(
                            outputs[oidx * self.cfg.seperate_head:(oidx + 1) * self.cfg.seperate_head, :, :],
                            outputs[ojdx * self.cfg.seperate_head:(ojdx + 1) * self.cfg.seperate_head, :,
                            :]))  # [10, 48, 100], [1, 48, 100]
                cdiv = torch.cat(cdiv, dim=-1).mean(dim=-1).mean()
                cbone, cminbone, cmaxbone = compute_bone_percent_error(datas[:, :, self.cfg.t_his].view(b, -1, 3),
                                                                       outputs.view(self.cfg.nk, -1, 3,
                                                                                    self.cfg.t_pred), self.cfg.parents)
                cangle = compute_angle_error(outputs, self.valid_angle, "h36m")

            diversity += cdiv
            ade += cade
            fde += cfde
            mmade += cmmade
            mmfde += cmmfde
            bone += cbone
            min_bone += cminbone
            max_bone += cmaxbone
            angle += cangle

            if epoch == -1:
                # print(
                #     "Test {} + {} > it {}: div {:.4f} | ade {:.4f} |  fde {:.4f}  |  mmade {:.4f} |  mmfde {:.4f}".format(
                #         all_z_p1.shape[0], 0, i, cdiv, cade, cfde, cmmade, cmmfde))
                print(
                    "Test {} + {} > it {}: div {:.4f} | ade {:.4f} |  fde {:.4f}  |  mmade {:.4f} |  mmfde {:.4f} |  bone {:.4f} |  [{:.4f}, {:.4f}] |  angle {:.4f}".format(
                        all_z_p1.shape[0], 0, i, cdiv, cade, cfde, cmmade, cmmfde, cbone, cminbone, cmaxbone, cangle))

            if draw:
                if i == draw_i:
                    bidx = 0

                    origin = datas[bidx:bidx + 1].reshape(1, -1, 3,
                                                          self.cfg.t_total).cpu().data.numpy()  # 1, 16, 3, 125
                    origin = np.concatenate(
                        (np.expand_dims(np.mean(origin[:, [0, 3], :, :], axis=1), axis=1), origin),
                        axis=1)  # # 1, 17, 3, 125
                    origin *= 1000

                    all_outputs = outputs.cpu().data.numpy().reshape(self.cfg.nk, -1, 3, self.cfg.t_pred)  # 10, 16, 3, 100
                    all_outputs = np.concatenate((np.expand_dims(np.mean(all_outputs[:, [0, 3], :, :], axis=1), axis=1), all_outputs),
                        axis=1)  # 10, 17, 3, 100
                    all_outputs *= 1000
                    all_outputs = np.concatenate((np.repeat(origin[:, :, :, :self.cfg.t_his],
                                                            repeats=self.cfg.nk, axis=0), all_outputs),
                                                 axis=-1)  # 10, 17, 3, 125

                    all_to_draw = np.concatenate((origin, all_outputs), axis=0)  # 1 + 10, 17, 3, 125
                    draw_acc = [acc for acc in range(0, all_to_draw.shape[-1], 5)]
                    all_to_draw = all_to_draw[:, :, :, draw_acc][:, :, [0, 2], :]

                    draw_multi_seqs_2d(all_to_draw, gt_cnt=1, t_his=5, I=self.cfg.I17_plot,
                                       J=self.cfg.J17_plot,
                                       LR=self.cfg.LR17_plot,
                                       full_path=os.path.join(self.cfg.ckpt_dir, "images", "sample",
                                                              f"test_epo{epoch}idx{draw_i}.png"))

        diversity /= (i + 1)
        ade /= (i + 1)
        fde /= (i + 1)
        mmade /= (i + 1)
        mmfde /= (i + 1)
        bone /= (i + 1)
        min_bone /= (i + 1)
        max_bone /= (i + 1)
        angle /= (i + 1)
        self.summary.add_scalar(f"Test/div", diversity, epoch)
        self.summary.add_scalar(f"Test/ade", ade, epoch)
        self.summary.add_scalar(f"Test/fde", fde, epoch)
        self.summary.add_scalar(f"Test/mmade", mmade, epoch)
        self.summary.add_scalar(f"Test/mmfde", mmfde, epoch)
        self.summary.add_scalar(f"Test/bone", bone, epoch)
        self.summary.add_scalar(f"Test/min_bone", min_bone, epoch)
        self.summary.add_scalar(f"Test/max_bone", max_bone, epoch)
        self.summary.add_scalar(f"Test/angle", angle, epoch)

        return diversity, ade, fde, mmade, mmfde, bone, min_bone, max_bone, angle

    def random_choose_preds(self):
        self.model.eval()
        dg = self.test_data.onebyone_generator()

        generator_len = len(self.test_data.similat_gt_like_dlow) if not self.is_debug else 90
        # random_idx = list(np.sort(np.random.choice(generator_len, 50)))

        # random_idx = [26, 456, 501, 562, 974,
        #               1112, 1173, 1239, 1517, 1624,
        #               1693, 1795, 2272, 2621, 2770,
        #               2973, 3081, 3496, 3551, 3935,
        #               3973, 4150, 4472, 4839, 4861]
        random_idx = [107, 168, 294, 343, 427, 562, 601, 669, 892, 1003, 1100, 1118, 1128, 1210, 1265, 1392, 1931, 2093,
                      2125, 2142, 2143, 2596, 2606, 2613, 2722, 2731, 2733, 2749, 2967, 3105, 3115, 3143, 3158, 3386,
                      3603, 3680, 3696, 3881, 4113, 4139, 4364, 4383, 4475, 4535, 4661, 4710, 4782, 4926, 4949, 5096]
        all_preds = []
        all_gts = []
        for i, datas in enumerate(dg):
            if i in random_idx:
                # b, 48, 125
                b, vc, t = datas.shape
                # similars = self.test_data.similat_gt_like_dlow[i]  # 0/n, 48, 100
                # if similars.shape[0] == 0:
                #     continue

                datas = torch.from_numpy(datas).float().cuda(device=self.cfg.device)
                # similars = torch.from_numpy(similars).float().cuda(device=self.cfg.device)

                with torch.no_grad():
                    padded_inputs = datas[:, :, list(range(self.cfg.t_his)) + [self.cfg.t_his - 1] * self.cfg.t_pred]
                    padded_inputs_dct = dct_transform_torch(padded_inputs, self.dct_m,
                                                            dct_n=self.cfg.dct_n)  # b, 48, 10
                    padded_inputs_dct = padded_inputs_dct.view(b, -1, 3 * self.cfg.dct_n)  # # b, 16, 3*10

                    # eps = torch.randn((b, self.cfg.z_dim), device=self.cfg.device)
                    # repeated_eps_1 = torch.repeat_interleave(eps, repeats=self.cfg.nk, dim=0)
                    # todo train generator eps 不共享
                    repeated_eps_1 = torch.randn((b * self.cfg.nk, self.cfg.z_dim), device=self.cfg.device)

                    logtics = torch.ones((b * self.cfg.nk, 1, self.cfg.base_num_p1), device=self.cfg.device) / self.cfg.base_num_p1  # b*h, 1, 10
                    many_weights = self._sample_weight_gumbel_softmax(logtics, temperature=self.cfg.temperature_p1)  # b*h, 1, 10
                    all_z_p1, all_mean_p1, all_logvar_p1 = self.model(condition=padded_inputs_dct,
                                                                      repeated_eps=repeated_eps_1,
                                                                      many_weights=many_weights,
                                                                      temperature=None,
                                                                      multi_modal_head=self.cfg.nk,
                                                                      mode="p1")  # b*(10), 128

                    all_z = all_z_p1
                    all_outs_dct = self.model_t1.inference(
                        condition=torch.repeat_interleave(padded_inputs_dct, repeats=self.cfg.nk, dim=0),
                        z=all_z)  # b*h, 16, 30
                    all_outs_dct = all_outs_dct.reshape(b * self.cfg.nk, -1, self.cfg.dct_n)  # b*h, 48, 10
                    outputs = reverse_dct_torch(all_outs_dct, self.i_dct_m, self.cfg.t_total)  # b*h, 48, 125
                    outputs = outputs.view(self.cfg.nk, -1, self.cfg.t_total)[:, :, self.cfg.t_his:]  # 50, 48, 100
                    all_preds.append(outputs.cpu().data.numpy())
                    all_gts.append(datas.cpu().data.numpy())

        all_preds = np.stack(all_preds, axis=0) # n, 50, 48, 125
        np.save(os.path.join(f"./supp_h36m_ours_preds{self.cfg.nk}.npy"), all_preds)

        all_gts = np.stack(all_gts, axis=0) # n, 1, 48, 125
        np.save(os.path.join(f"./supp_h36m_ours_gts{self.cfg.nk}.npy"), all_gts)


    def run(self):
        for epoch in range(self.start_epoch, self.cfg.epoch_t2 + 1):
            self.summary.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)

            average_allloss, average_adeerrors, average_mmadeerrors, average_hinges, average_kls_p1, average_limblen, average_angle = self.train(
                epoch, draw=False)
            self.scheduler.step()

            print(
                "Train --> Epoch {}: all {:.4f} | ades {:.4f} |  mmades {:.4f} |  hinges {:.4f} |  klsp1 {:.4f} | losshis {:.6f} | losslimb {:.6f} |  lossangle {:.6f}".format(
                    epoch, average_allloss, average_adeerrors, average_mmadeerrors, average_hinges, average_kls_p1, 0,
                    average_limblen, average_angle))


            if self.is_debug:
                test_interval = 1
            else:
                test_interval = 20

            if epoch % test_interval == 0:
                diversity, ade, fde, mmade, mmfde, bone, min_bone, max_bone, angle = self.eval(epoch=epoch, draw=False)
                # print("Test --+ epo {}: div {:.4f} | ade {:.4f} |  fde {:.4f}  | mmade {:.4f} |  mmfde {:.4f}".format(
                #     epoch,
                #     diversity,
                #     ade,
                #     fde,
                #     mmade,
                #     mmfde))
                print(
                    "Test --+ epo {}: div {:.4f} | ade {:.4f} |  fde {:.4f}  | mmade {:.4f} |  mmfde {:.4f} |  bone {:.4f} [{:.4f}, {:.4f}] |  angle {:.4f}".format(
                        epoch,
                        diversity,
                        ade,
                        fde,
                        mmade,
                        mmfde, bone, min_bone, max_bone, angle))

            if epoch % 50 == 0:
                self.save(
                    os.path.join(self.cfg.ckpt_dir, "models", '{}_err{:.4f}.pth'.format(epoch, average_hinges)),
                    epoch, average_hinges)
