# Coded By LEO XU
# At 2022/2/18 14:54
import os

import pandas as pd

DataPath = 'datasets'
FeaturePath = 'datasets'
sep = ','
inter_data = pd.read_csv(os.path.join(DataPath, 'ecommercedata_ho.csv'), usecols=[0, 2], delimiter=sep, header=None,
                         engine='python', keep_default_na=False)

# itemid_data = pd.read_csv(os.path.join(DataPath, 'ecommercedata_ho.csv'), usecols=[0, 1], delimiter=sep, header=0,
#                           engine='python')
# timestamp_data = pd.read_csv(os.path.join(DataPath, 'ecommercedata_ho.csv'), usecols=[0, 4], delimiter=sep, header=0,
#                              engine='python')
# userid_data = pd.read_csv(os.path.join(DataPath, 'ecommercedata_ho.csv'), usecols=[0, 6], delimiter=sep, header=0,
#                           engine='python')

print(inter_data)
# processed_data = pd.merge(userid_data, itemid_data, on='InvoiceNo', how='left')
# processed_data = pd.merge(processed_data, timestamp_data, on='InvoiceNo', how='left')
# processed_data = processed_data.loc[:, [1, 2, 3]]
# print(processed_data)