#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :cfa.py
# @Time      :2021/11/29 上午10:10
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com


import numpy as np
import torch
import torch.nn as nn
from torch.nn import MSELoss

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType, ModelType
import torch.nn.functional as F


class CFautoencoder(GeneralRecommender):
    input_type = InputType.POINTWISE
    # type = ModelType.TRADITIONAL

    def __init__(self, config, dataset):
        super(CFautoencoder, self).__init__(config, dataset)

        self.TAG_ID = config['TAG_ID_FIELD']
        self.rho = config['rho']
        self.beta = config['beta']


        # define layers and loss
        input_size = dataset.tag_num
        hid_layer_1, hid_layer_2 = config['hid_layer_1'], config['hid_layer_2']
        self.user_neighbors = config['user_neighbors']

        self.encoder = nn.Linear(input_size, hid_layer_1, bias=True)
        # self.decoder = nn.Linear(hid_layer_1, output_size, bias=True)
        self.activate = nn.Sigmoid()
        self.loss = MSELoss()

        # get user-tag metrix A item-tag metrix B
        self.user_item_matrix = dataset.create_src_tgt_matrix(dataset.inter_feat, self.USER_ID, self.ITEM_ID, is_weight=False)
        self.user_tag_matrix = dataset.create_src_tgt_matrix(dataset.inter_feat, self.USER_ID, self.TAG_ID, is_weight=False)
        self.user_item_matrix = self._get_weights_mat(self.user_item_matrix).to_dense()
        self.user_tag_matrix = self._get_weights_mat(self.user_tag_matrix).to_dense()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def _get_weights_mat(self, weight_mat):
        mat_data = torch.FloatTensor(weight_mat.data)
        mat_indices = torch.FloatTensor(np.vstack((weight_mat.row, weight_mat.col)))
        return torch.sparse_coo_tensor(mat_indices, mat_data, weight_mat.shape).to(self.device)

    def get_user_profiles(self, user):
        return self.activate(self.encoder(user))

    def forward(self, user, item, tag):
        pass


    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        user_tag_in = self.user_tag_matrix[user]
        hid_vector = self.activate(self.encoder(user_tag_in))
        user_tag_out = self.activate(F.linear(hid_vector, weight=self.encoder.weight.t()))
        # user_tag_out = self.decoder(self.hid_vector)
        loss = self.loss(user_tag_in, user_tag_out)
        reg_spares = self._regularization_spares(user_tag_in)
        return loss + reg_spares

    def _regularization_spares(self, user_tag_in):
        p_hat = self.activate(self.encoder(user_tag_in))
        p_hat = torch.mean(p_hat, dim=1)
        p = self.rho * torch.ones(p_hat.shape).to(self.device)
        reg = self.beta * F.kl_div(p, p_hat)
        return reg / user_tag_in.shape[0]

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_profiles(user)
        pass

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        sim_score, topk_idx = self._full_sort_similarity(user)
        all_user_e = self.user_item_matrix[topk_idx]
        score = torch.mul(sim_score.unsqueeze(2), all_user_e)
        score = torch.sum(score, dim=1).squeeze()
        return score.view(-1)

    def _full_sort_similarity(self, user):
        user_tag_in = self.user_tag_matrix[user]
        user_e = self.activate(self.encoder(user_tag_in))
        all_user_e = self.activate(self.encoder(self.user_tag_matrix))
        src_tgt_sim = torch.matmul(user_e, all_user_e.transpose(0, 1))
        src_user_e = torch.norm(user_e, dim=1).unsqueeze(1)
        tgt_user_e = torch.norm(all_user_e, dim=1).unsqueeze(0)
        normlize = torch.mm(src_user_e, tgt_user_e)
        score = src_tgt_sim / normlize
        sim_score, topk_idx = torch.topk(score, k=1 + self.user_neighbors)
        sim_score, topk_idx = sim_score[:, 1:], topk_idx[:, 1:]
        return sim_score, topk_idx

    def _similarity(self, src_user_e, tgt_user_e):
        return src_user_e * tgt_user_e / (torch.norm(src_user_e) * torch.norm(tgt_user_e))


