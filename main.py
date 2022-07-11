#!/usr/bin/env python
# encoding: utf-8
'''
@project : diverse_sampling
@file    : main.py
@author  : levon
@contact : levondang@163.com
@ide     : PyCharm
@time    : 2022-07-11 22:59
'''
# ****************************************************************************************************************
# *********************************************** Environments ***************************************************
# ****************************************************************************************************************

import numpy as np
import random
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seed_torch(seed=3450):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

seed_torch()


# ****************************************************************************************************************
# *********************************************** Main ***********************************************************
# ****************************************************************************************************************

import argparse
import pandas as pd
from pprint import pprint

from h36m.runs import RunCVAE as RunCVAEH36m
from h36m.runs import RunDiverseSampling as RunDiverseSamplingH36m
from humaneva.runs import RunCVAE as RunCVAEHumaneva
from humaneva.runs import RunDiverseSampling as RunDiverseSamplingHumaneva

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--exp_name', type=str, default="h36m_t2", help="h36m_t1 / h36m_t2 / humaneva_t1 / humaneva_t2")
parser.add_argument('--is_train', type=bool, default='', help="")
parser.add_argument('--is_load', type=bool, default='1', help="")
parser.add_argument('--is_debug', type=bool, default='1', help="")

parser.add_argument('--model_path', type=str, default="", help="")

args = parser.parse_args()

if args.exp_name == "h36m_t1":
    args.model_path = os.path.join(r"./ckpt/pretrained", "h36m_t1.pth")
    r = RunCVAEH36m(exp_name=args.exp_name, is_debug=args.is_debug, args=args, device="cuda:0", num_works=0)

elif args.exp_name == "h36m_t2":
    args.model_path = os.path.join(r"./ckpt/pretrained", "h36m_t2.pth")
    r = RunDiverseSamplingH36m(exp_name=args.exp_name, is_debug=args.is_debug, args=args, device="cuda:0", num_works=0)

elif args.exp_name == "humaneva_t1":
    args.model_path = os.path.join(r"./ckpt/pretrained", "humaneva_t1.pth")
    r = RunCVAEHumaneva(exp_name=args.exp_name, is_debug=args.is_debug, args=args, device="cuda:0", num_works=0)

elif args.exp_name == "humaneva_t2":
    args.model_path = os.path.join(r"./ckpt/pretrained", "humaneva_t2.pth")
    r = RunDiverseSamplingHumaneva(exp_name=args.exp_name, is_debug=args.is_debug, args=args, device="cuda:0", num_works=0)

else:
    print("wrong exp_name!")


if args.is_load:
    r.restore(args.model_path)

if args.is_train:
    r.run()
    # r.random_choose_25()

else:
    diversity, ade, fde, mmade, mmfde, bone, min_bone, max_bone, angle = r.eval(epoch=-1, draw=False)
    # print("\n Test -->  div {:.4f} -- ade {:.4f} --  fde {:.4f} --  mmade {:.4f} --  mmfde {:.4f} ".format(div,
    #                                                                                            ade,
    #                                                                                             fde,
    #                                                                                             mmade,
    #                                                                                            mmfde))
    print(
        "\n Test -->  div {:.4f} | ade {:.4f} |  fde {:.4f}  | mmade {:.4f} |  mmfde {:.4f} |  bone {:.4f} [{:.4f}, {:.4f}] |  angle {:.4f}".format(
            diversity,
            ade,
            fde,
            mmade,
            mmfde, bone, min_bone, max_bone, angle))
