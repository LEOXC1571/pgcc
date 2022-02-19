#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :dspr.py
# @Time      :2021/12/1 上午9:56
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com

import numpy as np
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
import torch.nn.functional as F

class DeepSimPersionalRec(GeneralRecommender):
    input_type = InputType.POINTWISE # 这个模型的负采样需要自己构造

    def __init__(self, config, dataset):
        super(DeepSimPersionalRec, self).__init__(config, dataset)

        self.TAG_ID = config['TAG_ID_FIELD']
        # load parameters info


        # define layers and loss
        self.hid_neurons = config['hid_neurons']
        self.layers = nn.ModuleList()
        self.activation = torch.tanh

        self.user_profiles = dataset.create_src_tgt_matrix(dataset.inter_feat, self.USER_ID, self.TAG_ID)
        self.item_profiles = dataset.create_src_tgt_matrix(dataset.inter_feat, self.ITEM_ID, self.TAG_ID)
        self.all_item = set(self.item_profiles.row)
        self.user_profiles = self._get_weights_mat(self.user_profiles).to_dense()
        self.item_profiles = self._get_weights_mat(self.item_profiles).to_dense()

        # input layer
        self.layers.append(
            nn.Linear(self.user_profiles.shape[1], self.hid_neurons[0])
        )
        for l in range(1, len(self.hid_neurons)):
            self.layers.append(nn.Linear(self.hid_neurons[l-1], self.hid_neurons[l]))
        self.loss = SimLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def _get_weights_mat(self, weight_mat):
        mat_data = torch.FloatTensor(weight_mat.data)
        mat_indices = torch.FloatTensor(np.vstack((weight_mat.row, weight_mat.col)))
        return torch.sparse_coo_tensor(mat_indices, mat_data, weight_mat.shape).to(self.device)

    def get_user_profiles(self, user):
        user_p = self.user_profiles[user]
        for layer in self.layers:
            user_p = layer(user_p)
            user_p = self.activation(user_p)
        return user_p

    def get_item_profiles(self, item):
        item_p = self.item_profiles[item]
        for layer in self.layers:
            item_p = layer(item_p)
            item_p = self.activation(item_p)
        return item_p

    def forward(self, user, item):
        user_p = self.get_user_profiles(user)
        item_p = self.get_item_profiles(item)
        return user_p, item_p

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction['label'].cpu().numpy()
        pos_inter_num = int(sum(label))
        user_p, item_p = self.forward(user, item)

        pos_user_p, neg_user_p = user_p[:pos_inter_num], user_p[pos_inter_num:] # 一个 batch 只有一个user 这里是为了方便实现 # 这样效率太低了 需要优化
        pos_item_p, neg_item_p = item_p[:pos_inter_num], item_p[pos_inter_num:] # item 中只有第一个是正样本，其余都是负采样得到

        neg_user_p = neg_user_p.view(-1, pos_inter_num, neg_user_p.shape[1])
        neg_item_p = neg_item_p.view(-1, pos_inter_num, neg_item_p.shape[1])

        pos_similarity = self._similarity(pos_user_p, pos_item_p)
        neg_similarity = self._similarity(neg_user_p, neg_item_p)
        loss = self.loss(pos_similarity, neg_similarity)
        return loss



    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_p = self.get_user_profiles(user)
        all_item_p = self.get_item_profiles(range(self.n_items))
        score = torch.matmul(user_p, all_item_p.transpose(0, 1))
        return score




    def _similarity(self, src, tgt):
        if len(src.shape) == 2:
            return torch.cosine_similarity(src, tgt)
        else:
            return torch.cosine_similarity(src, tgt, dim=2)




class SimLoss(nn.Module):
    """ SimLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(SimLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_sim, neg_sim):
        return - torch.mean(torch.log(torch.exp(pos_sim)/torch.sum(torch.exp(neg_sim), dim=0)))
