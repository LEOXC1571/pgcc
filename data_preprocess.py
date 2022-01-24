# Coded By LEO XU
# At 2022/1/21 13:09

import pandas as pd
import numpy as np
from datetime import date
import datetime as dt
import os
from features import GetItemFeature, GetBertFeature


DataPath = 'datasets'
FeaturePath = 'datasets'

raw_data = pd.read_csv(os.path.join(DataPath, 'ecommercedata_ho.csv'), header=0, keep_default_na=False)
# raw_data.colomns = ['invoice_no', 'stock_code', 'description', 'quantity', 'invoice_date', 'unit_price', 'customer_id',
#                     'country']
print('--------data read complete--------')

# item_data = GetItemFeature(raw_data)

bert_data = GetBertFeature(raw_data)

# print(item_data)
#item_data.to_csv(os.path.join(DataPath,'item_data.csv'),index=None)
#print('----------Item data processed----------')