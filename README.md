# pgcc

## roadmap
1. dataset
   1. processing to recbole files
2. user-item 
   1. mf-base models (bpr-mf)
   2. dnn-base models (wide & deep, deepmf) *
   3. graph-base models (light-gcn)
3. feature model
   1. wide & deep *
   2. deepmf
   3. xgboost *
4. graph base model
   1. KGAT *
   2. lightgcn *


0. 需要安装 transformers 这个库
1. StockCode 总共 3676 项，注意过滤
2. GetBertFeature 文档见函数注释， gpu 生成 大概 30s

2022/2/13
1. item/user 特征提取完毕

LGCN:
best result: 
{'best_valid_score': 0.1532, 'valid_score_bigger': True, 
'best_valid_result': {'recall@20': 0.1532, 'precision@20': 0.1051, 'hit@20': 0.6902, 'ndcg@20': 0.1755, 'mrr@20': 0.3548},
'test_result': {'recall@20': 0.1405, 'precision@20': 0.0933, 'hit@20': 0.6531, 'ndcg@20': 0.1651, 'mrr@20': 0.3452}}
 best valid : {'recall@20': 0.1146, 'precision@20': 0.084, 'hit@20': 0.6303, 'ndcg@20': 0.147, 'mrr@20': 0.3558}
 test result: {'recall@20': 0.1185, 'precision@20': 0.0796, 'hit@20': 0.6237, 'ndcg@20': 0.1491, 'mrr@20': 0.362}