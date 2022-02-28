#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :run.py
# @Time      :2021/11/8 下午7:56
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com


import argparse
import os
import logging
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders
from recbole.model.general_recommender import LightGCN, BPR
from recbole.utils import init_logger, get_trainer, init_seed, set_color

import statics
from datas.dataset import TagBasedDataset


def objective_run(config_dict=None, config_file_list=None, saved=True):
    model_config_str, dataset_config_str = config_file_list[1:]
    model_name = model_config_str.split('/')[-1].split('.')[0]
    dataset = dataset_config_str.split('/')[-1].split('.')[0]
    model_class = statics.model_name_map[model_name]
    config = Config(model=model_class, dataset=dataset, config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    logging.basicConfig(level=logging.ERROR)
    dataset = TagBasedDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = model_class(config, train_data.dataset).to(config['device']) # why train_data.dataset?
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=saved)
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def run(model=None, dataset=None, saved=False):
    current_path = os.path.dirname(os.path.realpath(__file__))
    # base config file
    overall_init_file = os.path.join(current_path, 'config/overall.yaml')
    # model config file
    model_init_file = os.path.join(current_path, 'config/model/' + model + '.yaml')
    # dataset config file
    dataset_init_file = os.path.join(current_path, 'config/dataset/' + dataset + '.yaml')

    config_file_list = [overall_init_file, model_init_file, dataset_init_file]  # env model dataset

    model_class = statics.model_name_map[model]
    # configurations initialization
    config = Config(model=model_class, dataset=dataset, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    # dataset = TagBasedDataset(config)
    if config['save_dataset']:
        dataset.save()
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset) # train just using part edge
    if config['save_dataloaders']:
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    # model loading and initialization
    model = model_class(config, train_data.dataset).to(config['device']) # get model
    # model = BPR(config, train_data.dataset).to(config['device'])

    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, action='store', help="model name")
    parser.add_argument("--dataset", type=str, action='store', help="dataset name")
    args, unknown = parser.parse_known_args()

    model_name = args.model
    dataset_name = args.dataset

    run(model_name, dataset_name)
