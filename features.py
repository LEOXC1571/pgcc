# Coded By LEO XU
# At 2022/1/21 15:54

import pandas as pd
import numpy as np
import os

def GetItemFeature(dataset):
    item = dataset[['InvoiceNo','StockCode','Quantity','InvoiceDate','UnitPrice','CustomerID']].copy()
    item_feature = item
    return item_feature