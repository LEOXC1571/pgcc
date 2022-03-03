# Coded By LEO XU
# At 2022/3/3 20:29

import pandas as pd
import os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


current_path = os.path.dirname(os.path.realpath(__file__))
user_feature_data = pd.read_csv(os.path.join(current_path, 'raw_datasets/user_data.csv'), header=0)
feature_col = user_feature_data.columns.tolist()[1:7]
feature = user_feature_data.copy()
customerid=LabelEncoder().fit_transform(feature['CustomerID'])
feature['CustomerID']=customerid

features1=preprocessing.StandardScaler().fit_transform(feature[feature_col])
features1=pd.DataFrame(features1)
features1['R'] = 0
features1['F'] = 0
features1['M'] = 0
features1['R'] = features1[4] - features1[5]
features1['F'] = features1[0] - features1[2]
features1['M'] = features1[1] - features1[3]
features2 = pd.concat([user_feature_data, features1], ignore_index=False, axis=1)
features3 = features2[['CustomerID', 'R', 'F', 'M']]
features3.to_csv(os.path.join(current_path, 'raw_datasets/user_feat.csv'),index=None)