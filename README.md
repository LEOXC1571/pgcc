# pgcc

## roadmap
1. dataset
   1. processing to recbole files
2. user-item 
   1. mf-base models (bpr-mf)
   2. dnn-base models (wide & deep, deepmf)
   3. graph-base models (light-gcn)
3. feature model
   1. wide & deep
   2. deepmf
   3. xgboost
4. graph base model
   1. KGAT
   2. lightgcn


0. 需要安装 transformers 这个库
1. StockCode 总共 3676 项，注意过滤
2. GetBertFeature 文档见函数注释， gpu 生成 大概 30s

2022/2/13
1. item/user 特征提取完毕