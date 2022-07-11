#!/usr/bin/env python
# encoding: utf-8
'''
@project : clarify_confusion
@file    : draw_pictures.py
@author  : Levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2021-11-27 14:03
'''

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw_pic_single(mydata, I, J, LR, full_path, color='b'):
    # 22, 3, XZY
    # I
    # J
    # LR

    # # # ****************************
    # # # todo dlow 不用调整 调整坐标，规范数据格式，：这里由于转换过来后本身应满足需求，不需要专门 revert_coordinate 或者交换坐标轴
    # mydata = mydata[:, [0, 2, 1]]
    # # # ****************************

    x = mydata[:, 0]
    y = mydata[:, 1]
    z = mydata[:, 2]

    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    ax.grid(False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-1500, 1500])
    ax.set_ylim3d([-1500, 1500])
    ax.set_zlim3d([-1500, 1500])

    my_x_ticks = np.arange(-1500, 1500, 500)
    my_y_ticks = np.arange(-1500, 1500, 500)
    my_z_ticks = np.arange(-1500, 1500, 500)
    ax.set_xticks(my_x_ticks)
    ax.set_yticks(my_y_ticks)
    ax.set_zticks(my_z_ticks)


    ax.scatter(x, y, z, c=color)

    # (250, 40, 40) #FA2828 红
    # (245, 125, 125) #F57D7D 粉
    # (11, 11, 11) #0B0B0B 黑色
    # (180, 180, 180) #B4B4B4 灰色

    if color == 'b':
        line_c = "#B4B4B4"
    else:
        line_c = "#F57D7D"

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([mydata[I[i], j], mydata[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=line_c if LR[i] else line_c)

    plt.savefig(full_path)
    plt.close()

def draw_pic_single_2d(mydata, I, J, LR, full_path):
    # 22, 3, XZY

    x = mydata[:, 0]
    y = mydata[:, 1]

    plt.figure(figsize=(6, 6))

    plt.scatter(x, y, c='r')

    # (250, 40, 40) #FA2828 红
    # (245, 125, 125) #F57D7D 粉
    # (11, 11, 11) #0B0B0B 黑色
    # (180, 180, 180) #B4B4B4 灰色

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y = [np.array([mydata[I[i], j], mydata[J[i], j]]) for j in range(2)]
        # ax.plot(x, y, z, lw=2, color='#FA2828' if LR[i] else '#F57D7D')
        # ax.plot(x, y, z, lw=2, color='#0B0B0B' if LR[i] else '#B4B4B4')
        plt.plot(x, y, lw=2, color='g' if LR[i] else 'b')

    plt.xlim((-800, 800))
    plt.ylim((-1500, 800))
    # 设置坐标轴名称
    plt.xlabel('x')
    plt.ylabel('y')
    # 设置坐标轴刻度
    my_x_ticks = np.arange(-1000, 1000, 200)
    my_y_ticks = np.arange(-1000, 1000, 200)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.grid(False)

    plt.savefig(full_path)
    plt.close(1)

def draw_pic_gt_pred(gt, pred, I, J, LR, full_path):
    # 22, 3, XZY

    # # # ****************************
    # # # # todo dlow 不用调整调整坐标，规范数据格式，：这里由于转换过来后本身应满足需求，不需要专门 revert_coordinate 或者交换坐标轴
    # gt = gt[:, [0, 2, 1]]
    # pred = pred[:, [0, 2, 1]]
    #
    # # # ****************************

    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    ax.grid(False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d([-1500, 1500])
    ax.set_ylim3d([-1500, 1500])
    ax.set_zlim3d([-1500, 1500])

    my_x_ticks = np.arange(-1500, 1500, 500)
    my_y_ticks = np.arange(-1500, 1500, 500)
    my_z_ticks = np.arange(-1500, 1500, 500)
    ax.set_xticks(my_x_ticks)
    ax.set_yticks(my_y_ticks)
    ax.set_zticks(my_z_ticks)


    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='k', linewidths=1)
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='r', linewidths=1)

    # (250, 40, 40) #FA2828 红
    # (245, 125, 125) #F57D7D 粉
    # (11, 11, 11) #0B0B0B 黑色
    # (180, 180, 180) #B4B4B4 灰色

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([gt[I[i], j], gt[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=1, color='#0B0B0B' if LR[i] else '#B4B4B4')
    for i in np.arange(len(I)):
        x, y, z = [np.array([pred[I[i], j], pred[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color='#FA2828' if LR[i] else '#F57D7D')

    plt.savefig(full_path)
    plt.close()

def draw_pic_gt_pred_2d(gt, pred, I, J, LR, full_path):
    # 22, 3, XZY

    plt.figure(figsize=(6, 6))

    plt.scatter(gt[:, 0], gt[:, 1], c='k', linewidths=1)
    plt.scatter(pred[:, 0], pred[:, 1], c='r', linewidths=1)

    # (250, 40, 40) #FA2828 红
    # (245, 125, 125) #F57D7D 粉
    # (11, 11, 11) #0B0B0B 黑色
    # (180, 180, 180) #B4B4B4 灰色

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y = [np.array([gt[I[i], j], gt[J[i], j]]) for j in range(2)]
        plt.plot(x, y, lw=1, color='#0B0B0B' if LR[i] else '#B4B4B4')
    for i in np.arange(len(I)):
        x, y = [np.array([pred[I[i], j], pred[J[i], j]]) for j in range(2)]
        plt.plot(x, y, lw=2, color='#FA2828' if LR[i] else '#F57D7D')

    plt.xlim((-800, 800))
    plt.ylim((-1500, 800))
    # 设置坐标轴名称
    plt.xlabel('x')
    plt.ylabel('y')
    # 设置坐标轴刻度
    my_x_ticks = np.arange(-1000, 1000, 200)
    my_y_ticks = np.arange(-1000, 1000, 200)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.grid(False)

    plt.savefig(full_path)
    plt.close(1)


def draw_multi_seqs_2d(seqs, gt_cnt=3, I=[], J=[], LR=[], t_his=25, full_path="", x_period=[-800, 800], y_period=[1000, 1000], z_period=[-1000, 1000]):
    n, v, c, t = seqs.shape  # n, 17, 2, 125

    gts = seqs[:gt_cnt]
    preds = seqs[gt_cnt:]

    # (250, 40, 40) #FA2828 红
    # (245, 125, 125) #F57D7D 粉
    # (11, 11, 11) #0B0B0B 黑色
    # (180, 180, 180) #B4B4B4 灰色
    # blue_left = (0, 0, 205) # #0000CD
    # blue_right = (100, 149, 237)  #6495ED

    plt.figure(figsize=(int((x_period[1]-x_period[0]) / 1000 * t), int((z_period[1]-z_period[0]) / 1000 * n)))  # 只有设置为相等的值，才能保证坐标轴等间隔不会变形
    plt.xlabel('x')
    plt.ylabel('z')

    # 设置坐标轴刻度
    plt.xlim(x_period[0], x_period[1] + ((t - 1) * (x_period[1] - x_period[0])))
    plt.ylim(z_period[0] - ((n - 1) * (z_period[1] - z_period[0])), z_period[1])

    my_x_ticks = np.arange(x_period[0], x_period[1] + (t - 1) * (x_period[1] - x_period[0]), 1000)
    my_y_ticks = np.arange(z_period[0] - ((n - 1) * (z_period[1] - z_period[0])), z_period[1], 1000)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    plt.grid(False)

    draw_cnt = 0
    for gtidx in range(gts.shape[0]):
        draw_gt = gts[gtidx]  # v, c, t
        for tidx in range(draw_gt.shape[-1]):
            pose = draw_gt[:, :, tidx]  # v, c
            pose[:, 0] = pose[:, 0] + (tidx * (x_period[1] - x_period[0]))
            pose[:, 1] = pose[:, 1] - (draw_cnt * (z_period[1] - z_period[0]))

            if tidx < t_his:
                # plt.scatter(pose[:, 0], pose[:, 1], c='k', linewidths=1)
                for i in np.arange(len(I)):
                    x, y = [np.array([pose[I[i], j], pose[J[i], j]]) for j in range(2)]
                    plt.plot(x, y, lw=1, color='#0B0B0B' if LR[i] else '#B4B4B4')
            else:
                # plt.scatter(pose[:, 0], pose[:, 1], c='b', linewidths=1)
                for i in np.arange(len(I)):
                    x, y = [np.array([pose[I[i], j], pose[J[i], j]]) for j in range(2)]
                    plt.plot(x, y, lw=1, color='#0000CD' if LR[i] else '#6495ED')

        draw_cnt += 1

    for predidx in range(preds.shape[0]):
        draw_pred = preds[predidx]  # v, c, t
        for tidx in range(draw_pred.shape[-1]):
            pose = draw_pred[:, :, tidx]  # v, c
            pose[:, 0] = pose[:, 0] + (tidx * (x_period[1] - x_period[0]))
            pose[:, 1] = pose[:, 1] - (draw_cnt * (z_period[1] - z_period[0]))

            if tidx < t_his:
                # plt.scatter(pose[:, 0], pose[:, 1], c='k', linewidths=1)
                for i in np.arange(len(I)):
                    x, y = [np.array([pose[I[i], j], pose[J[i], j]]) for j in range(2)]
                    plt.plot(x, y, lw=1, color='#0B0B0B' if LR[i] else '#B4B4B4')
            else:
                # plt.scatter(pose[:, 0], pose[:, 1], c='b', linewidths=1)
                for i in np.arange(len(I)):
                    x, y = [np.array([pose[I[i], j], pose[J[i], j]]) for j in range(2)]
                    plt.plot(x, y, lw=1, color='#FA2828' if LR[i] else '#F57D7D')

        draw_cnt += 1

    plt.savefig(full_path)
    plt.close(1)


if __name__ == "__main__":


    pass

