#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__init__.py
# @Time      :2021/11/20 下午4:25
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com

from .datasets_params import datasets_params
from .models_params import models_params
from models import *
from recbole.model.general_recommender import Pop
from recbole.trainer import Trainer

recbole_models = {
    'BPR',
    'Pop'
}

model_name_map = {
    'Pop': Pop,
    'BPR': BPR,
    'LGCN' : LightGCN,
    'NGCF': NGCF,
    'WGCN': WideGCN
}

trainers = {
    'BPR': Trainer,
    'BPR-T': Trainer,
}
