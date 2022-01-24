#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :utils.py
# @Time      :2022/1/24 上午9:43
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com

import os
import torch


def init_device(use_gpu: bool, gpu_id: str, args=None):
    if use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")