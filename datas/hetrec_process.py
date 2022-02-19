#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :hetrec_process.py
# @Time      :2021/11/4 下午7:50
# @Author    :Miracleyin
# @Mail      :miracleyin@live.com

import bz2
import csv
import json
import operator
import os
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from datas.base_dataset import BaseDataset


class HETRECProcess(BaseDataset):

    def __init__(self, input_path, output_path, dataset_name='movielens'):
        super(HETRECProcess, self).__init__(input_path, output_path)
        _least_k = {"delicious": 15, "lastfm": 5, "movielens": 5}
        _inter_file = {"delicious": 'user_taggedbookmarks-timestamps.dat',
                       "lastfm": 'user_taggedartists-timestamps.dat',
                       "movielens": 'user_taggedmovies-timestamps.dat'}
        self.dataset_name = dataset_name
        self._least_k = _least_k[dataset_name]

        self.inter_file = os.path.join(self.input_path, _inter_file[dataset_name])

        self.sep = "\t"

        # output file
        self.output_inter_file, self.output_item_file, self.output_user_file = self.get_output_files()

        self.inter_fields = {0: 'user_id:token',
                             1: 'item_id:token',
                             2: 'tag_id:token'}

    def load_inter_data(self):
        origin_data = pd.read_csv(self.inter_file, delimiter=self.sep, engine='python')
        origin_data = origin_data.iloc[:, :3]  # 取出 user_id item_id tag_id
        tag_data = origin_data['tagID'].value_counts()
        del_tag = list(tag_data[origin_data['tagID']] >= self._least_k) # todo ugly there how to better?
        origin_data = origin_data[del_tag]
        origin_data.reset_index(drop=True, inplace=True)
        return origin_data




if __name__ == '__main__':
    input_path = './dataset/hetrec2011-delicious-2k'
    output_path = './dataset/delicious'
    l = HETRECProcess(input_path, output_path, 'delicious')
    l.convert_inter()
