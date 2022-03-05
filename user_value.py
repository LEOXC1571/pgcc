# Coded By LEO XU
# At 2022/3/3 20:29

import pandas as pd
import os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


current_path = os.path.dirname(os.path.realpath(__file__))
user_feature_data = pd.read_csv(os.path.join(current_path, 'raw_datasets/user_feature_data.csv'), header=0)
feature_col = user_feature_data.columns.tolist()[1:17]
feature = user_feature_data.copy()
customerid=LabelEncoder().fit_transform(feature['CustomerID'])
feature['CustomerID']=customerid

features1=preprocessing.StandardScaler().fit_transform(feature[feature_col])
features1=pd.DataFrame(features1)
features2 = feature[['CustomerID']].copy()
features3 = pd.concat([features2, features1], ignore_index=False, axis=1)
features3.to_csv(os.path.join(current_path, 'raw_datasets/user_feat.csv'),index=None)
user_vector_data = user_feature_data[['CustomerID']].copy()
# user_vector_data['uv'] =



item_feature_data = pd.read_csv(os.path.join(current_path, 'raw_datasets/item_feature_data.csv'), header=0)
feature_col0 = item_feature_data.columns.tolist()[1:17]
feature0 = item_feature_data.copy()
features4 = preprocessing.MinMaxScaler().fit_transform(feature0[feature_col0])
features4 = pd.DataFrame(features4)
features5 = feature0[['StockCode']].copy()
features6 = pd.concat([features5, features4], ignore_index=False, axis=1)
features6.to_csv(os.path.join(current_path, 'raw_datasets/item_feat.csv'),index=None)