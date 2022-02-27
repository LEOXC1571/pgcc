#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :convert_run.py
# @Time      :2021/11/8 下午7:56
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com


import argparse
import os
from logging import getLogger

from recbole.config import Config
from recbole.data import data_preparation, save_split_dataloaders
from recbole.model.general_recommender import LightGCN, BPR
from recbole.utils import init_logger, get_trainer, init_seed, set_color
from recbole.data.dataset import Dataset
import statics
from datas.dataset import TagBasedDataset


def run_recbole(model=None, dataset=None, saved=False):
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
    # dataset = create_dataset(config)
    dataset = Dataset(config) # user_id:token, not user_id: token. more space here
    if config['save_dataset']:
        dataset.save()
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    if config['save_dataloaders']:
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    # model loading and initialization
    model = model_class(config, train_data.dataset).to(config['device'])
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
    parser.add_argument("--save", action='store_true', help="saved model path", default=False)
    args, unknown = parser.parse_known_args()

    model_name = args.model
    dataset_name = args.dataset
    save_flag = args.save

    run_recbole(model_name, dataset_name, save_flag)
