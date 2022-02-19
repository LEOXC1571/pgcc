#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :datasets_params.py
# @Time      :2021/11/20 下午4:22
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com

datasets_params = {
    'ml-100k': {
        'min_user_inter_num': 5,
        'min_item_inter_num': 5,
        'eval_setting': 'TO_RS,full',
        'metrics': ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': 20,
        'valid_metric': 'Recall@20',
        "split_ratio": [0.6, 0.2, 0.2],
        'eval_batch_size': 1000000
    },
    'movielens': {
        'load_col': {'inter': ['user_id', 'item_id', 'tag_id']},
        'eval_setting': 'TO_RS,full',
        'metrics': ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': [10, 20],
        'valid_metric': 'Recall@20',
        "split_ratio": [0.6, 0.2, 0.2],
        'eval_batch_size': 1000000
    },
    'lastfm': {
        'load_col': {'inter': ['user_id', 'item_id', 'tag_id']},
        'min_user_inter_num': 5,
        'min_item_inter_num': 5,
        'eval_setting': 'RO_RS,full',
        'metrics': ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': [10, 20],
        'valid_metric': 'Recall@20',
        "split_ratio": [0.6, 0.2, 0.2],
        'eval_batch_size': 1000000,
    },
    'delicious': {
        'load_col': {'inter': ['user_id', 'item_id', 'tag_id']},
        'min_user_inter_num': 5,
        'min_item_inter_num': 5,
        'eval_setting': 'RO_RS,full',
        'metrics': ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': [10, 20],
        'valid_metric': 'Recall@20',
        "split_ratio": [0.6, 0.2, 0.2],
        'eval_batch_size': 1000000,
    },
    'ecommerce': {
        'load_col': {'inter': [ 'item_id', 'timestamp', 'user_id' ],
                     'item': ['item_id', 'total_order', 'total_sales'],
                     'user': ['user_id', 'user_total_orders', 'user_total_pur']},
        'min_user_inter_num': 2,
        'min_item_inter_num': 2,
        'eval_setting': 'RO_RS,full',
        'metrics': ['Recall', 'NDCG', 'Hit', 'Precision'],
        'topk': [10, 20],
        'valid_metric': 'Recall@20',
        "split_ratio": [0.6, 0.2, 0.2],
        'eval_batch_size': 50000,
    }
}
