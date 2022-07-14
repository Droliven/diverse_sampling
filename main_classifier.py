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

from fid_acc import Evaluate_FID_ACC_H36m
from fid_acc import Evaluate_FID_ACC_Humaneva

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--exp_name', type=str, default="h36m_t2", help="h36m_t2 / humaneva_t2")

args = parser.parse_args()

if args.exp_name == "h36m_t2":
    r = Evaluate_FID_ACC_H36m()
    r.restore(os.path.join("./ckpt/pretrained", "h36m_t2.pth"))

elif args.exp_name == "humaneva_t2":
    r = Evaluate_FID_ACC_Humaneva()
    r.restore(os.path.join("./ckpt/pretrained", "humaneva_t2.pth"))

else:
    print("wrong exp_name!")


fid, acc = r.compute_fid_and_acc()
print("\n Test -->  fid {:.4f} -- acc {:.4f}".format(fid, acc))

