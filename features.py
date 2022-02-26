# Coded By LEO XU
# At 2022/1/21 15:54

import datetime as dt

import pandas as pd
from tqdm import tqdm

from utils.utils import init_device


def get_day_gap_before(s):
    date_received, dates = s.split('/')
    dates = dates.split(';')
    gaps = []
    for d in dates:
        # 将时间差转化为天数
        this_gap = (dt.date(int(date_received[0:4]), int(date_received[5:7]), int(date_received[8:10])) - dt.date(
            int(d[0:4]), int(d[5:7]), int(d[8:10]))).days
        if this_gap > 0:
            gaps.append(this_gap)
    if len(gaps) == 0:
        return 0
    else:
        return min(gaps)

def isrefund(pur):
    if pur >= 0:
        pur = 0
    elif pur < 0:
        pur = 1
    return pur

def GetItemFeature(dataset):
    item = dataset[['InvoiceNo', 'StockCode', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID']].copy()
    # read item feature
    item_id = item[['StockCode']].copy()
    item_id.drop_duplicates(inplace=True)

    # sort item feature: total sales/total quantity
    # total order 商品总订单量
    total_orders = item[['StockCode']].copy()
    total_orders['total_orders'] = 1
    total_orders = total_orders.groupby('StockCode').agg('sum').reset_index()

    # sales 商品销量
    total_sales = item[['StockCode', 'Quantity']].copy()
    # maximun/minimun/mean sales 单笔最大/最小/平均销量
    max_sales = total_sales.groupby('StockCode').agg('max').reset_index()
    max_sales.rename(columns={'Quantity': 'maximum_sales'}, inplace=True)
    min_sales = total_sales.groupby('StockCode').agg('min').reset_index()
    min_sales.rename(columns={'Quantity': 'minimum_sales'}, inplace=True)
    mean_sales = total_sales.groupby('StockCode').agg('mean').reset_index()
    mean_sales.rename(columns={'Quantity': 'mean_sales'}, inplace=True)
    median_sales = total_sales.groupby('StockCode').agg('median').reset_index()
    median_sales.rename(columns={'Quantity': 'median_sales'}, inplace=True)
    # total sales 商品总销量
    total_sales = total_sales.groupby('StockCode').agg({'Quantity': 'sum'}).reset_index()
    total_sales.columns = ['StockCode', 'total_sales']

    # sales_figure 商品销售额
    sales_figure = item[['StockCode', 'Quantity', 'UnitPrice']].copy()
    sales_figure['unit_sales'] = 0
    sales_figure['unit_sales'] = sales_figure['Quantity'] * sales_figure['UnitPrice']
    # maximum/minimum/mean sales figure 单笔最高/最低/平均成交额
    max_sales_figure = sales_figure.groupby('StockCode').agg({'unit_sales': 'max'}).reset_index()
    max_sales_figure.rename(columns={'unit_sales': 'maximum_figure'}, inplace=True)
    min_sales_figure = sales_figure.groupby('StockCode').agg({'unit_sales': 'min'}).reset_index()
    min_sales_figure.rename(columns={'unit_sales': 'minimum_figure'}, inplace=True)
    mean_sales_figure = sales_figure.groupby('StockCode').agg({'unit_sales': 'mean'}).reset_index()
    mean_sales_figure.rename(columns={'unit_sales': 'mean_figure'}, inplace=True)
    median_sales_figure = sales_figure.groupby('StockCode').agg({'unit_sales': 'median'}).reset_index()
    median_sales_figure.rename(columns={'unit_sales': 'median_figure'}, inplace=True)
    # sales figure 单品总成交额
    sales_figure = sales_figure.groupby('StockCode').agg({'unit_sales': 'sum'}).reset_index()
    sales_figure.rename(columns={'unit_sales': 'sales_figure'}, inplace=True)

    # customer_count 购买者人数
    item_cus = item[['StockCode', 'CustomerID']].copy()
    # max/mean/median order from one customer 最大/平均/中位复购次数
    ic_1 = item_cus.groupby(['StockCode', 'CustomerID']).size().reset_index()
    ic_1.columns = ['StockCode', 'CustomerID', 'count']
    max_order_cus = ic_1.groupby(['StockCode'])['count'].max().reset_index()
    max_order_cus.columns = ['StockCode', 'max_order_cus']
    mean_order_cus = ic_1.groupby(['StockCode'])['count'].mean().reset_index()
    mean_order_cus.columns = ['StockCode', 'mean_order_cus']
    median_order_cus = ic_1.groupby(['StockCode'])['count'].median().reset_index()
    median_order_cus.columns = ['StockCode', 'median_order_cus']
    # customer count 购买者人数
    item_cus.drop_duplicates(inplace=True)
    cus_count = item_cus.groupby('StockCode').agg({'CustomerID': 'count'}).reset_index()
    cus_count.rename(columns={'CustomerID': 'cus_count'}, inplace=True)

    # 退货率
    refund = item[['StockCode', 'Quantity']].copy()
    refund['absquantity'] = 0
    refund['absquantity'] = refund['Quantity'].abs()
    # refund['refund_quantity'] = 0
    refund1 = refund.groupby(['StockCode'])['absquantity'].sum().reset_index()
    refund1['refund_quantity'] = 0
    refund2 = refund.groupby(['StockCode'])['Quantity'].sum().reset_index()
    refund1['refund_quantity'] = (refund1['absquantity']-refund2['Quantity']) / 2
    # refund1['real_quantity'] = 0
    # refund1['real_quantity'] = refund2['Quantity']+refund1['refund_quantity']
    refund1['refund_ratio'] = refund1['refund_quantity'] / (refund1['absquantity'] - refund1['refund_quantity'])
    refund3 = refund1[['StockCode', 'refund_ratio']].copy()



    # order_gap 最大、最小购买间隔
    item_datetime = item[['StockCode', 'InvoiceDate']].copy()
    item_datetime.InvoiceDate = item_datetime.InvoiceDate.astype('str')
    item_datetime = item_datetime.groupby(['StockCode'])['InvoiceDate'].agg(lambda x: ';'.join(x)).reset_index()
    item_datetime.rename(columns={'InvoiceDate': 'date_list'}, inplace=True)
    item_dt1 = dataset[['StockCode', 'InvoiceDate']].copy()
    item_dt1 = pd.merge(item_dt1, item_datetime, on=['StockCode'], how='left')
    item_dt1['invoicedate_datelist'] = item_dt1.InvoiceDate.astype('str') + '/' + item_dt1.date_list
    item_dt2 = dataset[['StockCode', 'InvoiceDate']].copy()
    item_dt2['day_gap_before'] = item_dt1.invoicedate_datelist.apply(get_day_gap_before)
    # item_dt2['day_gap_after'] = item_dt1.invoicedate_datelist.apply(get_day_gap_after)
    max_gap = item_dt2.groupby('StockCode').agg({'day_gap_before': 'max'}).reset_index()
    max_gap.rename(columns={'day_gap_before': 'max_order_gap'}, inplace=True)
    min_gap = item_dt2.groupby('StockCode').agg({'day_gap_before': 'min'}).reset_index()
    min_gap.rename(columns={'day_gap_before': 'min_order_gap'}, inplace=True)


    #退货率
    # refund = dataset[['StockCode',]]



    #
    item_feature = pd.merge(item_id, total_orders, on='StockCode', how='left')
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
    item_feature = pd.merge(item_feature, refund3, on='StockCode', how='left')
    item_feature = pd.merge(item_feature, max_gap, on='StockCode', how='left')
    item_feature = pd.merge(item_feature, min_gap, on='StockCode', how='left')
    return item_feature


def GetBertFeature(dataset: pd.DataFrame)->pd.DataFrame:
    """
    get bert feature from StockCode Description, using default pretrain model.
    same StockCode have same Description'
    :param dataset:
    :return: pd.DataFrame {StockCode: np.array shape 768} shape = (3676, 4)
    """

    stock_df = pd.DataFrame(dataset, columns=['StockCode', 'Description'], index=None)
    stock_df = pd.DataFrame(stock_df.drop_duplicates(['StockCode']))

    from transformers import AutoTokenizer, AutoModel
    device = init_device(True, 0)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')  # netwrok problem
    bertmodel = AutoModel.from_pretrained('bert-base-cased').to(device)

    stock_df_desc = stock_df['Description']
    desc_embeddings = []
    for desc in tqdm(stock_df_desc):
        inputs = tokenizer(desc, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        res = bertmodel(**inputs)
        embedding = res.last_hidden_state[:, 0, :].detach().to('cpu')
        desc_embeddings.append(embedding.numpy().tolist()[0])

    desc_embeddings = pd.DataFrame({'bert-mbeddings': desc_embeddings})
    stock_df = pd.concat([stock_df, desc_embeddings], axis=1)
    return stock_df


def GetUserFeature(dataset):
    user = dataset[['InvoiceNo', 'StockCode', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID']].copy()
    # read item feature
    user_id = user[['CustomerID']].copy()
    user_id.drop_duplicates(inplace=True)

    # user order 用户总物品数和总订单量
    uto = user[['CustomerID','InvoiceNo']].copy()
    user_total_pur = uto.groupby('CustomerID').agg('count').reset_index()
    user_total_pur.rename(columns={'InvoiceNo': 'user_total_pur'}, inplace=True)
    uto.drop_duplicates(inplace=True)
    user_total_orders = uto.groupby('CustomerID').agg('count').reset_index()
    user_total_orders.rename(columns={'InvoiceNo': 'user_total_orders'}, inplace=True)

    #user pur figure 商品购买额
    pur_figure = user[['InvoiceNo', 'StockCode', 'CustomerID', 'Quantity', 'UnitPrice']].copy()
    pur_figure['unit_sales'] = 0
    pur_figure['unit_sales'] = pur_figure['Quantity'] * pur_figure['UnitPrice']
    pur_figure.groupby(['InvoiceNo', 'CustomerID']).agg({'unit_sales': 'sum'}).reset_index()
    # maximum/minimum/mean sales figure 单笔最高/最低/平均成交额
    max_pur_figure = pur_figure.groupby('CustomerID').agg({'unit_sales': 'max'}).reset_index()
    max_pur_figure.rename(columns={'unit_sales': 'maximum_figure'}, inplace=True)
    min_pur_figure = pur_figure.groupby('CustomerID').agg({'unit_sales': 'min'}).reset_index()
    min_pur_figure.rename(columns={'unit_sales': 'minimum_figure'}, inplace=True)
    mean_pur_figure = pur_figure.groupby('CustomerID').agg({'unit_sales': 'mean'}).reset_index()
    mean_pur_figure.rename(columns={'unit_sales': 'mean_figure'}, inplace=True)
    median_pur_figure = pur_figure.groupby('CustomerID').agg({'unit_sales': 'median'}).reset_index()
    median_pur_figure.rename(columns={'unit_sales': 'median_figure'}, inplace=True)
    # total purchase figure 个人总购买金额
    total_pur_figure = pur_figure.groupby('CustomerID').agg({'unit_sales': 'sum'}).reset_index()
    total_pur_figure.rename(columns={'unit_sales': 'total_pur_figure'}, inplace=True)

    #refund 退单：退单率，（下单金额-退单金额）/（下单次数-退单次数）
    refund = user[['InvoiceNo', 'StockCode', 'CustomerID', 'Quantity', 'UnitPrice']].copy()
    refund['unit_sales'] = 0
    refund['unit_sales'] = refund['Quantity'] * refund['UnitPrice']
    refund.groupby(['InvoiceNo', 'CustomerID']).agg({'unit_sales': 'sum'}).reset_index()
    #
    user_pur = refund.groupby('CustomerID').agg({'unit_sales': 'sum'}).reset_index()
    total_order = refund.groupby('CustomerID').agg({'InvoiceNo': 'count'}).reset_index()
    total_order.rename(columns={'InvoiceNo': 'total_order'}, inplace=True)
    #
    refund['is_refund'] = 0
    refund['is_refund'] = refund.unit_sales.apply(isrefund)
    refund['refund_figure'] = 0
    refund['refund_figure'] = refund['is_refund'] * refund['unit_sales']
    refund_count = refund.groupby('CustomerID').agg({'is_refund': 'sum'}).reset_index()
    refund_count.rename(columns={'is_refund': 'refund_count'}, inplace=True)
    refund_count = pd.merge(refund_count, total_order, on='CustomerID', how='left')
    refund_count['refund_rate'] = 0
    refund_count['refund_rate'] = refund_count['refund_count']/(refund_count['total_order']-refund_count['refund_count'])
    refund_count['v2'] = 0
    refund_count['v2'] = refund_count['total_order'] - (2 * refund_count['refund_count'])
    #
    # refund_figure = refund.groupby('CustomerID').agg({'refund_figure': 'sum'}).reset_index()
    refund_count = pd.merge(refund_count, total_pur_figure, on='CustomerID', how='left')
    # refund_figure['v1'] = 0
    # refund_figure['v1'] = refund_figure['total_pur_figure']-refund_figure['refund_figure']
    # refund_count = pd.merge(refund_count, refund_figure, on='CustomerID', how='left')
    refund_count['special_var'] = 0
    refund_count['special_var'] = refund_count['total_pur_figure']/refund_count['v2']
    refund_result = refund_count[['CustomerID', 'refund_rate', 'special_var']].copy()


    #recent_pur
    recent_pur = dataset[['InvoiceDate', 'CustomerID']].copy()
    rp1 = recent_pur.groupby('CustomerID').agg({'InvoiceDate': 'max'}).reset_index()
    rp1['recent_pur_gap'] = 0
    for i in range(rp1.shape[0]):
        rp1['recent_pur_gap'][i] = (dataset['InvoiceDate'][389160]-rp1['InvoiceDate'][i]).days
    rp2 = rp1[['CustomerID', 'recent_pur_gap']].copy()
    rp3 = recent_pur.groupby('CustomerID').agg({'InvoiceDate': 'min'}).reset_index()
    rp3['pur_gap'] = 0
    for j in range(rp3.shape[0]):
        rp3['pur_gap'][j] = (rp1['InvoiceDate'][j] - rp3['InvoiceDate'][j]).days
    rp4 = rp3[['CustomerID', 'pur_gap']].copy()



    user_feature = pd.merge(user_id, user_total_orders, on='CustomerID', how='left')
    user_feature = pd.merge(user_feature, user_total_pur, on='CustomerID', how='left')
    user_feature = pd.merge(user_feature, max_pur_figure, on='CustomerID', how='left')
    user_feature = pd.merge(user_feature, min_pur_figure, on='CustomerID', how='left')
    user_feature = pd.merge(user_feature, mean_pur_figure, on='CustomerID', how='left')
    user_feature = pd.merge(user_feature, median_pur_figure, on='CustomerID', how='left')
    user_feature = pd.merge(user_feature, total_pur_figure, on='CustomerID', how='left')
    user_feature = pd.merge(user_feature, refund_result, on='CustomerID', how='left')
    user_feature = pd.merge(user_feature, rp2, on='CustomerID', how='left')
    user_feature = pd.merge(user_feature, rp4, on='CustomerID', how='left')
    return user_feature
