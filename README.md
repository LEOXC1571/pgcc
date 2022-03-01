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

best params:  {'learning_rate': 0.001, 'n_layers': 2, 'reg_weight': 0.01, 'seed': 2020}
best result: 
{'best_valid_score': 0.1544, 'valid_score_bigger': True, 'best_valid_result': {'recall@20': 0.1544, 'precision@20': 0.1048, 'hit@20': 0.6947, 'ndcg@20': 0.1731, 'mrr@20': 0.3441}, 'test_result': {'recall@20': 0.1383, 'precision@20': 0.0915, 'hit@20': 0.6462, 'ndcg@20': 0.1572, 'mrr@20': 0.3201}} 

best params:  {'learning_rate': 0.001, 'n_layers': 2, 'reg_weight': 0.01, 'seed': 2021}
best result: 
{'best_valid_score': 0.1541, 'valid_score_bigger': True, 'best_valid_result': {'recall@20': 0.1541, 'precision@20': 0.1046, 'hit@20': 0.6932, 'ndcg@20': 0.1734, 'mrr@20': 0.3439}, 'test_result': {'recall@20': 0.1393, 'precision@20': 0.092, 'hit@20': 0.6505, 'ndcg@20': 0.1588, 'mrr@20': 0.3259}}

best params:  {'learning_rate': 0.0005, 'n_layers': 1, 'reg_weight': 0.01, 'seed': 2019}
best result: 
{'best_valid_score': 0.1548, 'valid_score_bigger': True, 'best_valid_result': {'recall@20': 0.1548, 'precision@20': 0.1052, 'hit@20': 0.6968, 'ndcg@20': 0.1762, 'mrr@20': 0.3553}, 'test_result': {'recall@20': 0.141, 'precision@20': 0.0937, 'hit@20': 0.651, 'ndcg@20': 0.1637, 'mrr@20': 0.3387}}

best params:  {'learning_rate': 0.0005, 'n_layers': 1, 'reg_weight': 0.01, 'seed': 2018}
best result: 
{'best_valid_score': 0.1541, 'valid_score_bigger': True, 'best_valid_result': {'recall@20': 0.1541, 'precision@20': 0.1063, 'hit@20': 0.6941, 'ndcg@20': 0.177, 'mrr@20': 0.352}, 'test_result': {'recall@20': 0.1406, 'precision@20': 0.0929, 'hit@20': 0.6516, 'ndcg@20': 0.1611, 'mrr@20': 0.3269}}
