#!/usr/bin/env python
# encoding: utf-8
'''
@project : gsps_reimplementation
@file    : run_cvae.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-12-03 16:18
'''

from ..datas import MaoweiGSPS_Dynamic_Seq_Humaneva, MaoweiGSPS_Dynamic_Seq_Humaneva_ExpandDataset_T1, draw_multi_seqs_2d, get_dct_matrix, dct_transform_torch, reverse_dct_torch
from ..nets import CVAE
from ..configs import ConfigCVAE
from losses import loss_recover_history_t1, loss_recons, loss_kl_normal,loss_limb_length_t1, loss_valid_angle_t1, compute_diversity, compute_ade, compute_fde, compute_mmade, compute_mmfde, \
    compute_bone_percent_error, compute_angle_error, compute_diversity_between_twopart

from torch.optim import Adam, lr_scheduler
import torch
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pprint import pprint
import random
import numpy as np
import json
import matplotlib.pyplot as plt
import pickle

class RunCVAE():
    def __init__(self, exp_name="base_cvaegcndct", device="cuda:0", num_works=0, is_debug=False, args=None):
        super(RunCVAE, self).__init__()

        self.is_debug = is_debug

        # 参数
        self.start_epoch = 1
        self.best_accuracy = 1e15

        self.cfg = ConfigCVAE(exp_name=exp_name, device=device, num_works=num_works)

        print("\n================== Configs =================")
        pprint(vars(self.cfg), indent=4)
        print("==========================================\n")

        save_dict = {"args": args.__dict__, "cfgs": self.cfg.__dict__}
        save_json = json.dumps(save_dict)

        with open(os.path.join(self.cfg.ckpt_dir, "config.json"), 'w', encoding='utf-8') as f:
            f.write(save_json)

        # 模型
        self.model = CVAE(node_n=self.cfg.node_n, hidden_dim=self.cfg.hidden_dim, z_dim=self.cfg.z_dim, dct_n=self.cfg.dct_n, dropout_rate=self.cfg.dropout_rate)

        if self.cfg.device != "cpu":
            self.model.cuda(self.cfg.device)

        print(">>> total params of {}: {:.8f}M\n".format(exp_name, sum(p.numel() for p in self.model.parameters()) / 1000000.0))

        ## dct
        self.dct_m, self.i_dct_m = get_dct_matrix(self.cfg.t_total)
        if self.cfg.device != "cpu":
            self.dct_m = torch.from_numpy(self.dct_m).float().cuda()
            self.i_dct_m = torch.from_numpy(self.i_dct_m).float().cuda()

        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.lr_t1)

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - self.cfg.epoch_fix_t1) / float(self.cfg.epoch_t1 - self.cfg.epoch_fix_t1 + 1)
            return lr_l

        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)

        # 数据
        # self.train_data = MaoweiGSPS_Dynamic_Seq_Humaneva(data_path=self.cfg.base_data_dir,
        #                                          similar_idx_path=self.cfg.similar_idx_path,
        #                                          similar_pool_path=self.cfg.similar_pool_path, t_his=self.cfg.t_his,
        #                                          t_pred=self.cfg.t_pred, similar_cnt=self.cfg.train_similar_cnt,
        #                                          dynamic_sub_len=self.cfg.sub_len_train,
        #                                          batch_size=self.cfg.train_batch_size,
        #                                          joint_used_17=self.cfg.joint_used, subjects=self.cfg.subjects,
        #                                          parents_17=self.cfg.parents,
        #                                          mode="train", multimodal_threshold=self.cfg.multimodal_threshold,
        #                                          is_debug=self.is_debug)


        self.train_data = MaoweiGSPS_Dynamic_Seq_Humaneva_ExpandDataset_T1(data_path=self.cfg.base_data_dir,
                                                          similar_idx_path=self.cfg.similar_idx_path,
                                                          similar_pool_path=self.cfg.similar_pool_path,
                                                          t_his=self.cfg.t_his,
                                                          t_pred=self.cfg.t_pred,
                                                          dynamic_sub_len=self.cfg.sub_len_train,
                                                          batch_size=self.cfg.train_batch_size,
                                                          joint_used_17=self.cfg.joint_used, subjects=self.cfg.subjects,
                                                          parents_17=self.cfg.parents,
                                                          mode="train",
                                                          multimodal_threshold=self.cfg.multimodal_threshold,
                                                          is_debug=self.is_debug)

        self.test_data = MaoweiGSPS_Dynamic_Seq_Humaneva(data_path=self.cfg.base_data_dir,
                                                similar_idx_path=self.cfg.similar_idx_path,
                                                similar_pool_path=self.cfg.similar_pool_path, t_his=self.cfg.t_his,
                                                t_pred=self.cfg.t_pred, similar_cnt=0,
                                                batch_size=self.cfg.test_batch_size,
                                                joint_used_17=self.cfg.joint_used, subjects=self.cfg.subjects,
                                                parents_17=self.cfg.parents,
                                                mode="test", multimodal_threshold=self.cfg.multimodal_threshold,
                                                is_debug=self.is_debug)
        self.test_data.get_test_similat_gt_like_dlow()

        self.valid_angle = pickle.load(open(self.cfg.valid_angle_path, "rb"))  # dict 13
        print(f"{'valid angle'} loaded from {self.cfg.valid_angle_path} !")


        self.summary = SummaryWriter(self.cfg.ckpt_dir)

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
        average_all_loss = 0
        average_recons_error = 0
        average_vec = 0
        average_kl = 0
        average_his = 0
        average_limblen = 0
        average_angle = 0

        dg = self.train_data.batch_generator()
        generator_len = self.cfg.sub_len_train // self.train_data.batch_size if not self.is_debug else 200 // self.train_data.batch_size
        draw_i = random.randint(0, generator_len - 1)

        for i, datas in enumerate(dg):
            # [b, 48, 125], [b, 10, 48, 125]
            b, vc, t = datas.shape
            # skip the last batch if only have one sample for batch_norm layers
            if b == 1:
                continue

            self.global_step = (epoch - 1) * generator_len + i + 1

            datas = torch.from_numpy(datas).float().cuda(device=self.cfg.device)
            with torch.no_grad():
                padded_inputs = datas[:, :, list(range(self.cfg.t_his)) + [self.cfg.t_his - 1] * self.cfg.t_pred]
                padded_inputs_dct = dct_transform_torch(padded_inputs, self.dct_m, dct_n=self.cfg.dct_n)  # b, 48, 10
                padded_inputs_dct = padded_inputs_dct.view(b, -1, 3 * self.cfg.dct_n)  # # b, 16, 3*10

                padded_gts_dct = dct_transform_torch(datas, self.dct_m, dct_n=self.cfg.dct_n)  # b, 48, 10
                padded_gts_dct = padded_gts_dct.view(b, -1, 3 * self.cfg.dct_n)  # # b, 16, 3*10

            out_dct, posterior_mean, posterior_logvar = self.model(condition=padded_inputs_dct, data=padded_gts_dct) # b, 16, 3*10
            out_dct = out_dct.reshape(b, -1, self.cfg.dct_n)  # b, 48, 10
            outputs = reverse_dct_torch(out_dct, self.i_dct_m, self.cfg.t_total)  # b, 48, 125

            # loss
            recons_error = loss_recons(gt=datas[:, :, self.cfg.t_his:], pred=outputs[:, :, self.cfg.t_his:])  # b, 48, 100
            vec = (datas[:, :, self.cfg.t_his-1:self.cfg.t_his] - outputs[:, :, self.cfg.t_his:self.cfg.t_his+1]).pow(2).sum() / b
            kl = loss_kl_normal(posterior_mean, posterior_logvar)

            recoverhis = loss_recover_history_t1(pred=outputs[:, :, :self.cfg.t_his], gt=datas[:, :, :self.cfg.t_his]) # 这里用 25 帧
            limblen_err = loss_limb_length_t1(outputs, datas, parent_17=self.cfg.parents)  # 这里用 125帧
            angle_err = loss_valid_angle_t1(outputs[:, :, self.cfg.t_his:], self.valid_angle, data="humaneva")

            all_loss = recons_error * self.cfg.t1_recons_weight \
                       + vec * self.cfg.t1_vec_weight \
                       + kl * self.cfg.t1_kl_weight \
                       + recoverhis + self.cfg.t1_recoverhis_weight \
                       + limblen_err * self.cfg.t1_limblen_weight

            if angle_err > 0:
                all_loss += angle_err * self.cfg.t1_angle_weight

            # all_loss = recons_error * self.cfg.t1_recons_weight + vec * self.cfg.t1_vec_weight + kl * self.cfg.t1_kl_weight  # 1:1000:0.1

            self.optimizer.zero_grad()
            all_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), max_norm=100)
            self.optimizer.step()

            average_all_loss += all_loss
            average_recons_error += recons_error
            average_vec += vec
            average_kl += kl
            average_his += recoverhis
            average_limblen += limblen_err
            average_angle += angle_err

            # 画图
            if draw:
                if i == draw_i:
                    bidx = 0
                    origin = datas[bidx:bidx + 1].detach().cpu().numpy()  # 1, 48, 125
                    origin = origin.reshape(1, -1, 3, self.cfg.t_total)  # 1, 16, 3, 125
                    origin = np.concatenate((np.expand_dims(np.mean(origin[:, [7, 10], :, :], axis=1), axis=1), origin),
                                            axis=1)  # # 1, 17, 3, 125
                    origin *= 1000

                    output = outputs[bidx:bidx + 1].reshape(1, -1, 3, self.cfg.t_total).detach().cpu().numpy()  # 1, 16, 3, 125
                    output = np.concatenate((np.expand_dims(np.mean(output[:, [7, 10], :, :], axis=1), axis=1), output),
                                            axis=1)  # # 1, 17, 3, 100
                    output *= 1000

                    all_to_draw = np.concatenate((origin, output), axis=0)
                    draw_acc = [acc for acc in range(0, all_to_draw.shape[-1], 5)]
                    all_to_draw = all_to_draw[:, :, :, draw_acc][:, :, [0, 2], :]

                    draw_multi_seqs_2d(all_to_draw, gt_cnt=1, t_his=5,
                                       I=self.cfg.I17_plot, J=self.cfg.J17_plot,
                                       LR=self.cfg.LR17_plot,
                                       full_path=os.path.join(self.cfg.ckpt_dir, "images",
                                                              f"train_epo{epoch}idx{draw_i}.png"))

        average_all_loss /= (i + 1)
        average_recons_error /= (i + 1)
        average_vec /=  (i + 1)
        average_kl /=  (i + 1)
        average_his /=  (i + 1)
        average_limblen /=  (i + 1)
        average_angle /=  (i + 1)

        self.summary.add_scalar(f"loss/averageall", average_all_loss, epoch)
        self.summary.add_scalar(f"loss/averagerecons", average_recons_error, epoch)
        self.summary.add_scalar(f"loss/averagevec", average_vec, epoch)
        self.summary.add_scalar(f"loss/averagekl", average_kl, epoch)
        self.summary.add_scalar(f"loss/averagerhis", average_his, epoch)
        self.summary.add_scalar(f"loss/averagelimblen", average_limblen, epoch)
        self.summary.add_scalar(f"loss/averageangle", average_angle, epoch)
        return average_all_loss, average_recons_error, average_kl, average_vec, average_his, average_limblen, average_angle

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
            z = torch.randn(b*self.cfg.nk, self.cfg.z_dim).cuda(device=self.cfg.device)
            with torch.no_grad():
                padded_inputs = datas[:, :, list(range(self.cfg.t_his)) + [self.cfg.t_his - 1] * self.cfg.t_pred]
                padded_inputs_dct = dct_transform_torch(padded_inputs, self.dct_m, dct_n=self.cfg.dct_n)  # b, 48, 10
                padded_inputs_dct = padded_inputs_dct.view(b, -1, 3 * self.cfg.dct_n)  # # b, 16, 3*10
                padded_inputs_dct = torch.repeat_interleave(padded_inputs_dct, repeats=self.cfg.nk, dim=0)

                out_dct = self.model.inference(condition=padded_inputs_dct, z=z)  # b*50, 16, 3*10
                out_dct = out_dct.reshape([b*self.cfg.nk, self.cfg.node_n * 3, -1])  # b*50, 48, 10
                outputs = reverse_dct_torch(out_dct, self.i_dct_m, self.cfg.t_total)[:, :, self.cfg.t_his:]  # 50, 48, 100

                cdiv = compute_diversity(outputs).mean()  # [10, 48, 100], [1, 48, 100]
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

                cade = compute_ade(outputs, datas[:, :, self.cfg.t_his:])
                cfde = compute_fde(outputs, datas[:, :, self.cfg.t_his:])
                cmmade = compute_mmade(outputs, datas[:, :, self.cfg.t_his:], similars)
                cmmfde = compute_mmfde(outputs, datas[:, :, self.cfg.t_his:], similars)
                cbone, cminbone, cmaxbone = 0, 0, 0 # compute_bone_percent_error(datas[:, :, self.cfg.t_his].view(b, -1, 3), outputs.view(self.cfg.nk, -1, 3, self.cfg.t_pred), self.cfg.parents)
                cangle = 0 #compute_angle_error(outputs, self.valid_angle, "humaneva")

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
                # print("Test > it {}: div {:.4f} | ade {:.4f} |  fde {:.4f} |  mmade {:.4f} |  mmfde {:.4f} ".format(i,
                #                                                                                                     cdiv,
                #                                                                                                     cade,
                #                                                                                                     cfde,
                #                                                                                                     cmmade,
                #                                                                                                     cmmfde))

                print(
                    "Test {} + {} > it {}: div {:.4f} | ade {:.4f} |  fde {:.4f}  |  mmade {:.4f} |  mmfde {:.4f} |  bone {:.4f} |  [{:.4f}, {:.4f}] |  angle {:.4f}".format(
                        self.cfg.nk, 0, i, cdiv, cade, cfde, cmmade, cmmfde, cbone, cminbone, cmaxbone, cangle))


            if draw:
                if i == draw_i:
                    bidx = 0

                    origin = datas[bidx:bidx + 1].reshape(1, -1, 3,
                                                          self.cfg.t_total).cpu().data.numpy()  # 1, 16, 3, 125
                    origin = np.concatenate(
                        (np.expand_dims(np.mean(origin[:, [7, 10], :, :], axis=1), axis=1), origin),
                        axis=1)  # # 1, 17, 3, 125
                    origin *= 1000

                    all_outputs = outputs.cpu().data.numpy().reshape(self.cfg.nk, -1, 3,
                                                                     self.cfg.t_pred)  # 10, 16, 3, 100
                    all_outputs = np.concatenate(
                        (np.expand_dims(np.mean(all_outputs[:, [7, 10], :, :], axis=1), axis=1), all_outputs),
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
        # random_idx = np.arange(generator_len)
        # np.random.shuffle(random_idx)
        # random_idx = list(np.sort(random_idx[:50]))
        random_idx = [1, 2, 17, 21, 28, 45, 47, 48, 52, 55, 57, 58, 61, 66, 69, 73, 74, 77, 88, 89, 90, 101, 106, 107, 108, 112, 115, 117, 119, 122, 124, 126, 130, 133, 140, 141, 143, 152, 160, 161, 163, 165, 169, 170, 175, 176, 179, 180, 182, 183]
        all_preds = []
        all_gts = []
        for i, datas in enumerate(dg):
            if i in random_idx:

                # b, 48, 125
                b, vc, t = datas.shape
                datas = torch.from_numpy(datas).float().cuda(device=self.cfg.device)
                z = torch.randn(b * self.cfg.nk, self.cfg.z_dim).cuda(device=self.cfg.device)
                with torch.no_grad():
                    padded_inputs = datas[:, :, list(range(self.cfg.t_his)) + [self.cfg.t_his - 1] * self.cfg.t_pred]
                    padded_inputs_dct = dct_transform_torch(padded_inputs, self.dct_m,
                                                            dct_n=self.cfg.dct_n)  # b, 48, 10
                    padded_inputs_dct = padded_inputs_dct.view(b, -1, 3 * self.cfg.dct_n)  # # b, 16, 3*10
                    padded_inputs_dct = torch.repeat_interleave(padded_inputs_dct, repeats=self.cfg.nk, dim=0)

                    out_dct = self.model.inference(condition=padded_inputs_dct, z=z)  # b*50, 16, 3*10
                    out_dct = out_dct.reshape([b * self.cfg.nk, self.cfg.node_n * 3, -1])  # b*50, 48, 10
                    outputs = reverse_dct_torch(out_dct, self.i_dct_m, self.cfg.t_total)[:, :, self.cfg.t_his:]  # 50, 48, 100

                    all_preds.append(outputs.cpu().data.numpy())
                    all_gts.append(datas.cpu().data.numpy())

        all_preds = np.stack(all_preds, axis=0)
        np.save(os.path.join(f"./supp_humaneva_cvae_preds{self.cfg.nk}.npy"), all_preds)  # n, 50, 48, 100

        all_gts = np.stack(all_gts, axis=0)  # n, 1, 48, 125
        np.save(os.path.join(f"./supp_humaneva_cvae_gts{self.cfg.nk}.npy"), all_gts)



    def run(self):
        for epoch in range(self.start_epoch, self.cfg.epoch_t1 + 1):
            self.summary.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)

            average_all_loss, average_recons_error, average_kl, average_vec, average_his, average_limblen, average_angle = self.train(epoch, draw=False)
            self.scheduler.step()

            print(
                "train >>> epoch {}: all {:.6f},  lossrecons {:.6f}, losskl {:.6f}, lossvec {:.6f}, losshis {:.6f}, losslimb {:.6f}, lossangle {:.6f}".format(epoch, average_all_loss, average_recons_error, average_kl, average_vec, average_his, average_limblen, average_angle))

            if self.is_debug:
                test_interval = 1
            else:
                test_interval = 20
            if epoch % test_interval == 0:
                diversity, ade, fde, mmade, mmfde, bone, min_bone, max_bone, angle = self.eval(epoch=epoch, draw=True if epoch % 50 == 0 else False)
                # print("Test > epo {}: div {:.4f} | ade {:.4f} |  fde {:.4f} |  mmade {:.4f} |  mmfde {:.4f} ".format(
                #     epoch,
                #     div,
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
                        os.path.join(self.cfg.ckpt_dir, "models", '{}_err{:.4f}.pth'.format(epoch, average_all_loss)),
                        epoch, average_all_loss)



