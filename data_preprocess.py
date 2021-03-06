# Coded By LEO XU
# At 2022/1/21 13:09

import os
import datetime as dt
import pandas as pd
from features import GetItemFeature, GetBertFeature, GetUserFeature

DataPath = 'raw_datasets'
FeaturePath = 'raw_datasets'

raw_data = pd.read_csv(os.path.join(DataPath, 'ecommercedata_ho.csv'), header=0, keep_default_na=False)
# processed_data = pd.read_csv(os.path.join(DataPath, 'ecommercedata_processed.csv'), header=0, keep_default_na=False)

raw_data['InvoiceDate'] = pd.to_datetime(raw_data['InvoiceDate'])
# order_date = pd.to_datetime(raw_data['InvoiceDate'],format='%Y/%m/%d')
order_date = raw_data['InvoiceDate'].dt.strftime('%Y%m%d')
order_time = raw_data['InvoiceDate'].dt.strftime('%H-%M')
order_day = raw_data['InvoiceDate'].dt.strftime('%w')
raw_data = pd.concat([raw_data, order_date, order_time, order_day], ignore_index=True, axis=1)
raw_data.columns = ['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID',
                    'Country', 'order_date', 'order_time', 'order_day']
# day = (raw_data['InvoiceDate'][100000]-raw_data['InvoiceDate'][1]).days
# print(day)
print('--------data read complete--------')

# item_data = GetItemFeature(raw_data)
user_data = GetUserFeature(raw_data)
# bert_data = GetBertFeature(raw_data)
# inter_data = raw_data[['StockCode', 'order_date', 'CustomerID']]
# print(item_data)

# item_data.to_csv(os.path.join(DataPath,'item_data.csv'),index=None)
user_data.to_csv(os.path.join(DataPath,'user_data.csv'),index=None)
# item_des_data = pd.merge(item_data,bert_data,on='StockCode',how='left')
# item_des_data.to_csv(os.path.join(DataPath,'item_des_data.csv'),index=None)
# inter_data.to_csv(os.path.join(DataPath,'inter_data.csv'),index=None)
print('----------Item data processed----------')
