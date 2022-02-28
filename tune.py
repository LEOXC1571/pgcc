#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :tune.py
# @Time      :2021/11/20 下午7:42
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com
import argparse
import os
from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function
from recbole.utils import init_logger, ensure_dir, get_local_time
from rec_run import objective_run
import multiprocessing


def tune(model=None, dataset=None, params_name=None):
    from itertools import product
    current_path = os.path.dirname(os.path.realpath(__file__))
    # base config file
    overall_init_file = os.path.join(current_path, 'config/overall.yaml')
    # model config file
    model_init_file = os.path.join(current_path, 'config/model/' + model + '.yaml')
    # dataset config file
    dataset_init_file = os.path.join(current_path, 'config/dataset/' + dataset + '.yaml')

    config_file_list = [overall_init_file, model_init_file, dataset_init_file]  # env model dataset

    # hyper tune space
    params_space = os.path.join(current_path, 'config/params/' + params_name + '.params')


    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    hp = HyperTuning(objective_run, algo='exhaustive',
                     params_file=params_space, fixed_config_file_list=config_file_list)
    hp.run()
    ensure_dir('hyper_experiment_1_18') # 自己手动记录一下实验开始的日期
    hp.export_result(output_file=f'hyper_experiment_1_18/{model}-{dataset}.result')
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, action='store', help="model name")
    parser.add_argument("--dataset", type=str, action='store', help="dataset name")
    parser.add_argument('--params', type=str, default=None, help='parameters file')
    args, unknown = parser.parse_known_args()

    model_name = args.model
    dataset_name = args.dataset
    params_name = args.params

    tune(model_name, dataset_name, params_name)