#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :predict.py
# @Time      :2022/3/8 上午10:56
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com

import argparse
import os
import logging
from logging import getLogger
import pandas as pd
import numpy as np
import torch
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
from recbole.model.general_recommender import bpr, pop
from recbole.data.interaction import Interaction
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color
import statics

def load_data_and_model(model_name, model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config['seed'], config['reproducibility'])
    model_class = statics.model_name_map[model_name]
    model = model_class(config, train_data.dataset).to(config['device'])  # get model
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))

    return config, model, dataset, train_data, valid_data, test_data


def predict(model_name, model_path, data_path):
    # pop model
    pop_config, pop_model, pop_dataset, pop_train_data, pop_valid_data, pop_test_data = load_data_and_model(
        model_name='Pop',
        model_file='saved/Pop-Mar-08-2022_13-58-26.pth',
    )  # Here you can replace it by your model path.

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_name=model_name,
        model_file=model_path,
    )  # Here you can replace it by your model path.

    pred_data = pd.read_csv(data_path)
    pred_array = pred_data['CustomerID'].values

    user_preditem = {}  # key:id value item
    for id in pred_array:
        id = str(id)
        if id not in dataset.field2token_id[dataset.uid_field]:  # 如果不在，那么使用pop预测
            token = dataset.token2id(dataset.uid_field, ['12347'])
            topk_score, topk_iid_list = full_sort_topk(token, pop_model, pop_test_data, k=20, device=config['device'])
            external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
            user_preditem[id] = external_item_list
        else:  # 在 那么使用 model 预测
            token = dataset.token2id(dataset.uid_field, [id])
            token = torch.tensor(token)
            input_interaction = dataset.join(Interaction({dataset.uid_field: token}))
            input_interaction = input_interaction.to(config['device'])
            try:
                scores = model.full_sort_predict(input_interaction)
            except NotImplementedError:
                input_interaction = input_interaction.repeat_interleave(dataset.item_num)
                input_interaction.update(
                    test_data.dataset.get_item_feature().to(config['device']).repeat(len(token)))
                scores = model.predict(input_interaction)
            scores = scores.view(-1, dataset.item_num)
            scores[:, 0] = -np.inf  # set scores of [pad] to -inf
            topk_score, topk_iid_list = torch.topk(scores, 20)
            external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
            user_preditem[id] = external_item_list

    pred_items = []
    for id in pred_array:
        pred_items.append(user_preditem[str(id)][0])

    pred_user = np.array(pred_array).reshape(-1, 1)
    pred_item = np.array(pred_items)
    res = np.hstack((pred_user, pred_item))
    col = ['CustomerID'] + [str(i + 1) for i in range(20)]
    res = pd.DataFrame(res, columns=col)
    res.to_csv('./res.csv', index=False)


if __name__ == '__main__':

    model_name = 'WGCN'
    model_path = 'saved/WideGCN-Mar-08-2022_15-10-44.pth'  # pth not the least one is the base
    data_path = 'raw_datasets/ecommercedata_pre.csv'

    predict(model_name, model_path, data_path)
