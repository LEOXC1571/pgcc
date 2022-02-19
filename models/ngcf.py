#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :ngcf.py
# @Time      :2022/1/7 下午2:28
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import BiGNNLayer, SparseDropout
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class NGCF(GeneralRecommender):
    r"""NGCF is a model that incorporate GNN for recommendation.
    We implement the model following the original author with a pairwise training mode.
    """
    input_type = InputType.PAIRWISE
    def __init__(self, config, dataset):
        super(NGCF, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # load parameters info
