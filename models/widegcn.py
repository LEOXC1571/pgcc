#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :lightgcn.py
# @Time      :2021/12/30 下午9:24
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization, xavier_normal_, constant_
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType, FeatureSource, FeatureType
from recbole.model.layers import FMEmbedding, FMFirstOrderLinear, MLPLayers
from recbole.model.general_recommender import lightgcn


class WideGCN(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(WideGCN, self).__init__(config, dataset)

        # load user, item feat info
        self.field_names = dataset.fields(
            source=[
                FeatureSource.USER,
                FeatureSource.ITEM,
            ]
        )

        # build double_tower
        self.double_tower = config['double_tower']
        if self.double_tower is None:
            self.double_tower = False

        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []
        self.num_feature_field = 0

        if self.double_tower:  # get user feat item feat nums
            self.user_field_names = dataset.fields(source=[FeatureSource.USER, FeatureSource.USER_ID])
            self.item_field_names = dataset.fields(source=[FeatureSource.ITEM, FeatureSource.ITEM_ID])
            self.field_names = self.user_field_names + self.item_field_names
            self.user_token_field_num = 0
            self.user_float_field_num = 0
            self.user_token_seq_field_num = 0
            for field_name in self.user_field_names:
                if field_name == 'user_id':
                    continue
                if dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.user_token_field_num += 1
                elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.user_token_seq_field_num += 1
                else:
                    self.user_float_field_num += dataset.num(field_name)

            self.item_token_field_num = 0
            self.item_float_field_num = 0
            self.item_token_seq_field_num = 0
            for field_name in self.item_field_names:
                if field_name == 'item_id':
                    continue
                if field_name[:3] == 'neg':
                    continue
                if dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.item_token_field_num += 1
                elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.item_token_seq_field_num += 1
                else:
                    self.item_float_field_num += dataset.num(field_name)

        # count feat nums
        for field_name in self.field_names:
            if field_name[:3] == 'neg':
                continue
            if field_name[-2:] == 'id':
                continue
            if dataset.field2type[field_name] == FeatureType.TOKEN:
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            else:
                self.float_field_names.append(field_name)
                self.float_field_dims.append(dataset.num(field_name))
            self.num_feature_field += 1

        # load parameters info
        self.fm_embedding_size = config['fm_embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']

        self.mf_embedding_szie = config['mf_embedding_size']
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN

        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        # define layers and loss
        if len(self.token_field_dims) > 0:  # 相当于使用 lgcn 取代这一部分
            self.token_field_offsets = np.array((0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long)
            self.token_embedding_table = FMEmbedding(  #
                self.token_field_dims, self.token_field_offsets, self.lr_embedding_size
            )

        if len(self.float_field_dims) > 0:
            self.float_embedding_table = nn.Embedding(  # 32 * 4
                np.sum(self.float_field_dims, dtype=np.int32), self.fm_embedding_size
                # 32 * 4 = 128 -> user 64 * 1 | item 64 * 1
            )

        if len(self.token_seq_field_dims) > 0:
            self.token_seq_embedding_table = nn.ModuleList()
            for token_seq_field_dim in self.token_seq_field_dims:
                self.token_seq_embedding_table.append(nn.Embedding(token_seq_field_dim, self.fm_embedding_size))

        self.u_mf_embedding = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.mf_embedding_szie)
        self.i_mf_embedding = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.mf_embedding_szie)

        size_list = [self.fm_embedding_size * self.num_feature_field // 2] + self.mlp_hidden_size
        self.user_mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.item_mlp_layers = MLPLayers(size_list, self.dropout_prob)
        self.deep_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        # self.apply(xavier_uniform_initialization)
        self.apply(self._init_weights)
        # self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def get_norm_adj_mat(self):
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        user_embeddings = self.u_mf_embedding.weight
        item_embeddings = self.i_mf_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self, interaction):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_mf_embeddings, item_all_mf_embeddings = torch.split(lightgcn_all_embeddings,
                                                                     [self.n_users, self.n_items])
        user_batch_fm_embeddings, item_batch_fm_embeddings = self.concat_embed_input_fields(interaction)

        return user_all_mf_embeddings, item_all_mf_embeddings, user_batch_fm_embeddings, item_batch_fm_embeddings

    def concat_embed_input_fields(self, interaction):
        sparse_embedding, dense_embedding = self.double_tower_embed_input_fields(interaction)
        user_dense, item_dense = dense_embedding
        #     if not self.double_tower:
        #         sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        #     else:
        #         sparse_embedding, dense_embedding = self.double_tower_embed_input_fields(interaction)
        #
        #     all_embeddings = []
        #     if isinstance(sparse_embedding, tuple):
        #         for sp in sparse_embedding:
        #             if sp is not None:
        #                 all_embeddings.append(sp)
        #         for de in dense_embedding:
        #             if de is not None:
        #                 all_embeddings.append(de)
        #     else:
        #         if sparse_embedding is not None:
        #             all_embeddings.append(sparse_embedding)
        #         if dense_embedding is not None and len(dense_embedding.shape) == 3:
        #             all_embeddings.append(dense_embedding)
        #
        #     # cat cf wd
        #     return torch.cat(all_embeddings, dim=1)  # [batch_size, num_field, embed_dim]
        #
        return user_dense, item_dense

    def embed_input_fields(self, interaction):
        """Embed the whole feature columns.

        Args:
            interaction (Interaction): The input data collection.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns.
            torch.FloatTensor: The embedding tensor of float sequence columns.
        """
        float_fields = []
        for field_name in self.float_field_names:
            if len(interaction[field_name].shape) == 2:
                float_fields.append(interaction[field_name])
            else:
                float_fields.append(interaction[field_name].unsqueeze(1))
        if len(float_fields) > 0:
            float_fields = torch.cat(float_fields, dim=1)  # [batch_size, num_float_field]
        float_fields_embedding = self.embed_float_fields(float_fields)  # float_fields 512 32

        token_fields = []  # may be concat lgn?
        for field_name in self.token_field_names:
            token_fields.append(interaction[field_name].unsqueeze(1))
        if len(token_fields) > 0:
            token_fields = torch.cat(token_fields, dim=1)  # [batch_size, num_token_field]
        else:
            token_fields = None
        # [batch_size, num_token_field, embed_dim] or None
        token_fields_embedding = self.embed_token_fields(token_fields)

        token_seq_fields = []
        for field_name in self.token_seq_field_names:
            token_seq_fields.append(interaction[field_name])
        # [batch_size, num_token_seq_field, embed_dim] or None
        token_seq_fields_embedding = self.embed_token_seq_fields(token_seq_fields)

        if token_fields_embedding is None:
            sparse_embedding = token_seq_fields_embedding
        else:
            if token_seq_fields_embedding is None:
                sparse_embedding = token_fields_embedding
            else:
                sparse_embedding = torch.cat([token_fields_embedding, token_seq_fields_embedding], dim=1)

        dense_embedding = float_fields_embedding

        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding

    def double_tower_embed_input_fields(self, interaction):
        """Embed the whole feature columns in a double tower way.

        Args:
            interaction (Interaction): The input data collection.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of token sequence columns in the second part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the second part.

        """
        if not self.double_tower:
            raise RuntimeError('Please check your model hyper parameters and set \'double tower\' as True')
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        if dense_embedding is not None:
            first_dense_embedding, second_dense_embedding = \
                torch.split(dense_embedding, [self.user_float_field_num, self.item_float_field_num], dim=1)
        else:
            first_dense_embedding, second_dense_embedding = None, None

        if sparse_embedding is not None:
            sizes = [
                self.user_token_seq_field_num, self.item_token_seq_field_num, self.user_token_field_num,
                self.item_token_field_num
            ]
            first_token_seq_embedding, second_token_seq_embedding, first_token_embedding, second_token_embedding = \
                torch.split(sparse_embedding, sizes, dim=1)
            first_sparse_embedding = torch.cat([first_token_seq_embedding, first_token_embedding], dim=1)
            second_sparse_embedding = torch.cat([second_token_seq_embedding, second_token_embedding], dim=1)
        else:
            first_sparse_embedding, second_sparse_embedding = None, None

        return (first_sparse_embedding, second_sparse_embedding), (first_dense_embedding, second_dense_embedding)

    def embed_float_fields(self, float_fields, embed=True):
        """Embed the float feature columns

        Args:
            float_fields (torch.FloatTensor): The input dense tensor. shape of [batch_size, num_float_field]
            embed (bool): Return the embedding of columns or just the columns itself. Defaults to ``True``.

        Returns:
            torch.FloatTensor: The result embedding tensor of float columns.
        """
        # input Tensor shape : [batch_size, num_float_field]
        if not embed or float_fields is None:
            return float_fields

        num_float_field = float_fields.shape[1]
        # [batch_size, num_float_field]
        index = torch.arange(0, num_float_field).unsqueeze(0).expand_as(float_fields).long().to(self.device)

        # [batch_size, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table(index)
        float_embedding = torch.mul(float_embedding, float_fields.unsqueeze(2))

        return float_embedding

    def embed_token_fields(self, token_fields):
        """Embed the token feature columns

        Args:
            token_fields (torch.LongTensor): The input tensor. shape of [batch_size, num_token_field]

        Returns:
            torch.FloatTensor: The result embedding tensor of token columns.
        """
        # input Tensor shape : [batch_size, num_token_field]
        if token_fields is None:
            return None
        # [batch_size, num_token_field, embed_dim]
        token_embedding = self.token_embedding_table(token_fields)

        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields, mode='mean'):
        """Embed the token feature columns

        Args:
            token_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]
            mode (str): How to aggregate the embedding of feature in this field. default=mean

        Returns:
            torch.FloatTensor: The result embedding tensor of token sequence columns.
        """
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[i]
            mask = token_seq_field != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            token_seq_embedding = embedding_table(token_seq_field)  # [batch_size, seq_len, embed_dim]

            mask = mask.unsqueeze(2).expand_as(token_seq_embedding)  # [batch_size, seq_len, embed_dim]
            if mode == 'max':
                masked_token_seq_embedding = token_seq_embedding - (1 - mask) * 1e9  # [batch_size, seq_len, embed_dim]
                result = torch.max(masked_token_seq_embedding, dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
            elif mode == 'sum':
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
            else:
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=1)  # [batch_size, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, embed_dim]
                result = result.unsqueeze(1)  # [batch_size, 1, embed_dim]
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(fields_result, dim=1)  # [batch_size, num_token_seq_field, embed_dim]

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        # if self.restore_user_mfe is not None or self.restore_item_mfe is not None:
        #     self.restore_user_mfe, self.restore_item_mfe = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_all_mf_embeddings, item_all_mf_embeddings, user_batch_fm_embeddings, item_batch_fm_embeddings = self.forward(
            interaction)

        user_batch_mf_embeddings = user_all_mf_embeddings[user]
        posi_batch_mf_embeddings = item_all_mf_embeddings[pos_item]
        negi_batch_mf_embeddings = item_all_mf_embeddings[neg_item]

        u_deep_output = self.user_mlp_layers(user_batch_fm_embeddings.view(-1, 64))
        pos_deep_output = self.item_mlp_layers(item_batch_fm_embeddings.view(-1, 64))
        neg_deep_output = self.item_mlp_layers(item_batch_fm_embeddings.view(-1, 64))

        # u_embeddings = torch.cat([u_deep_output, user_batch_mf_embeddings], dim=1)
        # pos_embeddings = torch.cat([pos_deep_output, posi_batch_mf_embeddings], dim=1)
        # neg_embeddings = torch.cat([neg_deep_output, negi_batch_mf_embeddings], dim=1)
        u_embeddings = torch.mean(torch.stack([u_deep_output, user_batch_mf_embeddings], dim=2),dim=2)
        pos_embeddings = torch.mean(torch.stack([pos_deep_output, posi_batch_mf_embeddings], dim=2),dim=2)
        neg_embeddings = torch.mean(torch.stack([neg_deep_output, negi_batch_mf_embeddings], dim=2),dim=2)

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.u_mf_embedding(user)
        pos_ego_embeddings = self.i_mf_embedding(pos_item)
        neg_ego_embeddings = self.i_mf_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_mf_embeddings, item_all_mf_embeddings, user_batch_fm_embeddings, item_batch_fm_embeddings = self.forward(
            interaction)
        user_batch_mf_embeddings = user_all_mf_embeddings[user]
        item_batch_mf_embeddings = item_all_mf_embeddings[item]
        u_deep_output = self.user_mlp_layers(user_batch_fm_embeddings.view(-1, 64))
        i_deep_output = self.item_mlp_layers(item_batch_fm_embeddings.view(-1, 64))

        u_embeddings = torch.mean(torch.stack([u_deep_output, user_batch_mf_embeddings], dim=2),dim=2)
        i_embeddings = torch.mean(torch.stack([i_deep_output, item_batch_mf_embeddings], dim=2),dim=2)

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    # def full_sort_predict(self, interaction):
    #     user = interaction[self.USER_ID]
    #     if self.restore_user_mfe is None or self.restore_item_mfe is None:
    #         self.restore_user_mfe, self.restore_item_mfe, self.restore_user_fme, self.restore_item_fme = self.forward(interaction)
    #     # get user embedding from storage variable
    #     u_embeddings = self.restore_user_e[user]
    #     # dot with all item embedding to accelerate
    #     scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))
    #
    #     return scores.view(-1)
