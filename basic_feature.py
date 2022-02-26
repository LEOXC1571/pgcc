# Coded By LEO XU
# At 2022/2/25 13:29

import pandas as pd
import os

DataPath = 'raw_datasets'
FeaturePath = 'raw_datasets'

raw_data = pd.read_csv(os.path.join(DataPath, 'ecommercedata_ho.csv'), header=0, keep_default_na=False)

# order_number = raw_data.groupby(['CustomerID']).count().reset_index()
# print(order_number)

data = raw_data.copy()
data['unit_sales'] = 0
data['unit_sales'] = data['Quantity'] * data['UnitPrice']
total_figure = data.agg({'unit_sales': 'sum'})
print(total_figure)