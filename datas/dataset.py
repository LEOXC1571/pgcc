#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :dataset.py
# @Time      :2021/11/4 下午9:56
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com
import time

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from collections import Counter, defaultdict
from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction
from recbole.utils.enum_type import FeatureType
from recbole.utils import FeatureSource, FeatureType, get_local_time, set_color


class TagBasedDataset(Dataset):
    """:class:`TagBasedDataset` is based on :`~recbole.data.dataset.dataset.Dataset`,
    and load_col 'tag_id' additionally

    tag assisgment [``user_id``, ``item_id``, ``tag_id``]

    Attributes:
        tid_field (str): The same as ``config['TAG_ID_FIELD']``

    """

    def __init__(self, config):
        super().__init__(config)

    def _get_field_from_config(self):
        super()._get_field_from_config()

        self.tid_field = self.config['TAG_ID_FIELD']
        self._check_field('tid_field')
        if self.tid_field is None:
            raise ValueError(
                'Tag need to be set at the same time or not set at the same time.'
            )
        self.logger.debug(set_color('tid_field', 'blue') + f': {self.tid_field}')

    # def _data_filtering(self):
    #     super()._data_filtering()
    #     self._filter_tag()

    # def _filter_tag(self):
    #     pass

    def _load_data(self, token, dataset_path):
        super()._load_data(token, dataset_path)

    # def _build_feat_name_list(self):
    #     feat_name_list = super()._build_feat_name_list()

    def _data_processing(self):
        """Data preprocessing, including:

        - Data filtering
        - Remap ID
        - Missing value imputation
        - Normalization
        - Preloading weights initialization
        """
        super()._data_processing()
        # self.user_tag_value_matrix = self.create_src_tgt_matrix(self.inter_feat, self.uid_field, self.tid_field)
        # self.item_tag_value_matrix = self.create_src_tgt_matrix(self.inter_feat, self.iid_field, self.tid_field)
        # self.user_tag_matrix = self._create_ui_tag_matrix(self.inter_feat, self.uid_field, self.tid_field, form='coo', is_weight=False)
        # self.item_tag_matrix = self._create_ui_tag_matrix(self.inter_feat, self.iid_field, self.tid_field, form='coo', is_weight=False)

    def __str__(self):
        info = [
            super().__str__()
        ]
        if self.tid_field:
            info.extend([
                set_color('The number of tags', 'blue') + f': {self.tag_num}',
                set_color('Average actions of tags', 'blue') + f': {self.avg_actions_of_tags}'
            ])
        return '\n'.join(info)

    # def _build_feat_name_list(self):
    #     feat_name_list = super()._build_feat_name_list()
    #     if self.tid_field is not None:
    #         feat_name_list.append('tag_feat')
    #     return feat_name_list

    def _init_alias(self):
        """Add :attr:`alias_of_tag_id` and update :attr:`_rest_fields`.
        """
        self._set_alias('tag_id', [self.tid_field])
        super()._init_alias()

    def create_src_tgt_matrix(self, df_feat, source_field, target_field, is_weight=True):
        """Get sparse matrix that describe relations between two fields.

        Source and target should be token-like fields.

        Sparse matrix has shape (``self.num(source_field)``, ``self.num(target_field)``).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = df_feat[value_field][src, tgt]``.

        Args:
            df_feat (Interaction): Feature where src and tgt exist.
            source_field (str): Source field
            target_field (str): Target field
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not isinstance(df_feat, pd.DataFrame):
            try:
                df_feat = pd.DataFrame.from_dict(df_feat.interaction)
            except BaseException:
                raise ValueError(f'feat from is not supported.')
        df_feat = df_feat.groupby([source_field, target_field]).size()
        df_feat.name = 'weights'
        df_feat = df_feat.reset_index()
        src = df_feat[source_field]
        tgt = df_feat[target_field]
        if is_weight:
            data = df_feat['weights']
        else:
            data = np.ones(len(df_feat))
        mat = coo_matrix((data, (src, tgt)), shape=(self.num(source_field), self.num(target_field)))
        return mat
        # if form == 'coo':
        #     return mat
        # elif form == 'csr':
        #     return mat.tocsr()
        # else:
        #     raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented.')








    @property
    def tag_num(self):
        """Get the number of different tokens of tags.

       Returns:
           int: Number of different tokens of tags.
       """
        return self.num(self.tid_field)

    @property
    def avg_actions_of_tags(self):
        """Get the average number of tags' interaction records.

        Returns:
             numpy.float64: Average number of tags' interaction records.
        """
        if isinstance(self.inter_feat, pd.DataFrame):
            return np.mean(self.inter_feat.groupby(self.tid_field).size())
        else:
            return np.mean(list(Counter(self.inter_feat[self.tid_field].numpy()).values()))
