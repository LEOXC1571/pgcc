# Coded By LEO XU
# At 2022/1/21 15:54

import pandas as pd
import numpy as np
import os
import datetime as dt
from datetime import date

def GetItemFeature(dataset):
    item = dataset[['InvoiceNo','StockCode','Quantity','InvoiceDate','UnitPrice','CustomerID']].copy()
    #read item feature
    item_id = item[['StockCode']].copy()
    item_id.drop_duplicates(inplace=True)

    #sort item feature: total sales/total quantity
    #total order 商品总订单量
    total_orders = item[['StockCode']].copy()
    total_orders['total_orders'] = 1
    total_orders = total_orders.groupby('StockCode').agg('sum').reset_index()

    #sales 商品销量
    total_sales = item[['StockCode','Quantity']].copy()
    # maximun/minimun/mean sales 单笔最大/最小/平均销量
    max_sales = total_sales.groupby('StockCode').agg('max').reset_index()
    max_sales.rename(columns={'Quantity': 'maximum_sales'}, inplace=True)
    min_sales = total_sales.groupby('StockCode').agg('min').reset_index()
    min_sales.rename(columns={'Quantity': 'minimum_sales'}, inplace=True)
    mean_sales = total_sales.groupby('StockCode').agg('mean').reset_index()
    mean_sales.rename(columns={'Quantity': 'mean_sales'}, inplace=True)
    median_sales = total_sales.groupby('StockCode').agg('median').reset_index()
    median_sales.rename(columns={'Quantity': 'median_sales'}, inplace=True)
    #total sales 商品总销量
    total_sales = total_sales.groupby('StockCode').agg({'Quantity':'sum'}).reset_index()
    total_sales.columns = ['StockCode','total_sales']


    #sales_figure 商品销售额
    sales_figure = item[['StockCode','Quantity','UnitPrice']].copy()
    sales_figure['unit_sales'] = 0
    sales_figure['unit_sales'] = sales_figure['Quantity'] * sales_figure['UnitPrice']
    #maximum/minimum/mean sales figure 单笔最高/最低/平均成交额
    max_sales_figure = sales_figure.groupby('StockCode').agg({'unit_sales':'max'}).reset_index()
    max_sales_figure.rename(columns={'unit_sales': 'maximum_figure'}, inplace=True)
    min_sales_figure = sales_figure.groupby('StockCode').agg({'unit_sales':'min'}).reset_index()
    min_sales_figure.rename(columns={'unit_sales': 'minimum_figure'}, inplace=True)
    mean_sales_figure = sales_figure.groupby('StockCode').agg({'unit_sales':'mean'}).reset_index()
    mean_sales_figure.rename(columns={'unit_sales': 'mean_figure'}, inplace=True)
    median_sales_figure = sales_figure.groupby('StockCode').agg({'unit_sales':'median'}).reset_index()
    median_sales_figure.rename(columns={'unit_sales': 'median_figure'}, inplace=True)
    #sales figure 单品总成交额
    sales_figure = sales_figure.groupby('StockCode').agg({'unit_sales':'sum'}).reset_index()
    sales_figure.rename(columns={'unit_sales': 'sales_figure'}, inplace=True)

    #customer_count 购买者人数
    item_cus = item[['StockCode','CustomerID']].copy()
    #max/mean/median order from one customer 最大/平均/中位复购次数
    ic_1 = item_cus.groupby(['StockCode', 'CustomerID']).size().reset_index()
    ic_1.columns = ['StockCode', 'CustomerID', 'count']
    max_order_cus = ic_1.groupby(['StockCode'])['count'].max().reset_index()
    max_order_cus.columns = ['StockCode', 'max_order_cus']
    mean_order_cus = ic_1.groupby(['StockCode'])['count'].mean().reset_index()
    mean_order_cus.columns = ['StockCode', 'mean_order_cus']
    median_order_cus = ic_1.groupby(['StockCode'])['count'].median().reset_index()
    median_order_cus.columns = ['StockCode', 'median_order_cus']
    #customer count 购买者人数
    item_cus.drop_duplicates(inplace=True)
    cus_count = item_cus.groupby('StockCode').agg({'CustomerID':'count'}).reset_index()
    cus_count.rename(columns={'CustomerID':'cus_count'},inplace=True)

    #单笔再打



    #
    #
    item_feature = pd.merge(item_id, total_orders, on='StockCode',how='left')
    item_feature = pd.merge(item_feature, total_sales, on='StockCode', how='left')
    item_feature = pd.merge(item_feature, max_sales, on='StockCode', how='left')
    item_feature = pd.merge(item_feature, min_sales, on='StockCode', how='left')
    item_feature = pd.merge(item_feature, mean_sales, on='StockCode', how='left')
    item_feature = pd.merge(item_feature, median_sales, on='StockCode', how='left')
    item_feature = pd.merge(item_feature, sales_figure, on='StockCode', how='left')
    item_feature = pd.merge(item_feature, max_sales_figure, on='StockCode', how='left')
    item_feature = pd.merge(item_feature, min_sales_figure, on='StockCode', how='left')
    item_feature = pd.merge(item_feature, mean_sales_figure, on='StockCode', how='left')
    item_feature = pd.merge(item_feature, median_sales_figure, on='StockCode', how='left')
    item_feature = pd.merge(item_feature, cus_count, on='StockCode', how='left')
    item_feature = pd.merge(item_feature, max_order_cus, on='StockCode', how='left')
    item_feature = pd.merge(item_feature, mean_order_cus, on='StockCode', how='left')
    item_feature = pd.merge(item_feature, median_order_cus, on='StockCode', how='left')
    # item_feature = min_sales_figure
    return item_feature
