# Coded By LEO XU
# At 2022/2/25 19:03

import math
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import LabelEncoder


# import mglearn


def kmo(dataset_corr):
    corr_inv = np.linalg.inv(dataset_corr)
    nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
    A = np.ones((nrow_inv_corr, ncol_inv_corr))
    for i in range(0, nrow_inv_corr, 1):
        for j in range(i, ncol_inv_corr, 1):
            A[i, j] = -(corr_inv[i, j]) / (math.sqrt(corr_inv[i, i] * corr_inv[j, j]))
            A[j, i] = A[i, j]
    dataset_corr = np.asarray(dataset_corr)
    kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))
    kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
    kmo_value = kmo_num / kmo_denom
    return kmo_value



current_path = os.path.dirname(os.path.realpath(__file__))
item_feature_data = pd.read_csv(os.path.join(current_path, 'raw_datasets/item_feature_data.csv'), header=0)
feature_col = item_feature_data.columns.tolist()[1:17]
feat_corr = item_feature_data[feature_col].corr()
feat_kmo = kmo(feat_corr)

sns.set_style("white")
plt.figure(figsize = (10,10))
# np.zero_like的意思就是生成一个和你所给数组a相同shape的全0数组。
mask = np.zeros_like(feat_corr, dtype=bool)
# np.triu_indices_from()返回方阵的上三角矩阵的索引
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax = sns.heatmap(feat_corr, mask=mask, cmap=cmap, square=True, annot=True,fmt='0.2f')
ax.set_title("feat_corr Variables Relation")
# plt.show()

feature = item_feature_data.copy()
stockcode=LabelEncoder().fit_transform(feature['StockCode'])
feature['StockCode']=stockcode

features1=preprocessing.MinMaxScaler().fit_transform(feature[feature_col])
features1=pd.DataFrame(features1)
features1.head()
# features1.to_csv(os.path.join(current_path, 'outputs/tables/feat_minmaxscaler.csv'),index=None)
#对数据集进行PCA降维（信息保留为99.99%）
pca=PCA(n_components=0.9999)  #保证降维后的数据保持90%的信息，则填0.9
features2=pca.fit_transform(features1)
#降维后，每个主要成分的解释方差占比（解释PC携带的信息多少）
ratio=pca.explained_variance_ratio_
print('各主成分的解释方差占比：',ratio)
#降维后有几个成分
print('降维后有几个成分：',len(ratio))
#累计解释方差占比
cum_ratio=np.cumsum(ratio)
print('累计解释方差占比：',cum_ratio)
#绘制PCA降维后各成分方差占比的直方图和累计方差占比折线图
plt.figure(figsize=(8,6))
X=range(1,len(ratio)+1)
Y=ratio
plt.bar(X,Y,edgecolor='black')
plt.plot(X,Y,'r.-')
plt.plot(X,cum_ratio,'b.-')
plt.ylabel('explained_variance_ratio')
plt.xlabel('PCA')
# plt.show()
#PCA选择降维保留3个主要成分
pca=PCA(n_components=3)
features3=pca.fit_transform(features1)
pca_feat = pd.concat([item_feature_data, pd.DataFrame(features3)], ignore_index=False, axis=1)
pca_feat.rename(columns={0: 'pca1'}, inplace=True)
pca_feat.rename(columns={1: 'pca2'}, inplace=True)
pca_feat.rename(columns={2: 'pca3'}, inplace=True)
pca_feat[['StockCode', 'pca1', 'pca2', 'pca3']].to_csv(os.path.join(current_path, 'outputs/tables/feat_pca3.csv'),index=None)
#降维后的累计各成分方差占比和（即解释PC携带的信息多少）
print(sum(pca.explained_variance_ratio_))
factor_load = pca.components_.T * np.sqrt(pca.explained_variance_)
factor_load = pd.DataFrame(factor_load).T
# factor_load.to_csv(os.path.join(current_path, 'outputs/tables/factor_load.csv'),index=None)
##肘方法看k值，簇内离差平方和
# 对每一个k值进行聚类并且记下对于的SSE，然后画出k和SSE的关系图
sse = []
for i in range(1, 15):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    km.fit(features3)
    sse.append(km.inertia_)

plt.plot(range(1, 15), sse, marker='*')
plt.xlabel('n_clusters')
plt.ylabel('distortions')
plt.title("The Elbow Method")
# plt.show()


fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
X=range(1,len(ratio)+1)
Y=ratio
ax1.bar(X,Y,edgecolor='black')
ax1.plot(X,Y,'r.-')
ax1.plot(X,cum_ratio,'b.-')
ax1.set_ylabel('explained_variance_ratio')
ax1.set_xlabel('PCA')
ax1.set_title("Explained Variance Ratio")
ax2.plot(range(1, 15), sse, marker='*')
ax2.set_xlabel('number of clusters')
ax2.set_ylabel('distortions')
ax2.set_title("The Elbow Method")
# plt.title('K-Means PCA 3D')
plt.savefig('pca-kmeans-em.png')
plt.show()

sns.set_style("white")
plt.figure(figsize = (10,4))
# np.zero_like的意思就是生成一个和你所给数组a相同shape的全0数组。
mask = np.zeros_like(factor_load, dtype=bool)
# np.triu_indices_from()返回方阵的上三角矩阵的索引
# mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 20, as_cmap=True)
ax = sns.heatmap(factor_load, mask=mask, cmap=cmap, square=True, annot=True,fmt='0.3f')
ax.set_title("Factor load of 3 principle components")
plt.savefig('factor-load.png')
plt.show()
















# 进行K-Means聚类分析
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
kmeans.fit(features3)
lab = kmeans.predict(features3)
lab = lab.T
lab_result = pd.DataFrame(lab)
item_feature_data = pd.concat([item_feature_data, lab_result], ignore_index=False, axis=1)
item_feature_data.rename(columns={0: 'kmeans_pre'}, inplace=True)
# item_feature_data.to_csv(os.path.join(current_path, 'outputs/tables/kmeans_pre.csv'),index=None)
item_feature_data[['StockCode', 'kmeans_pre']].to_csv(os.path.join(current_path, 'outputs/tables/kmeans_pre.csv'),index=None)
#绘制聚类结果2维的散点图
plt.figure(figsize=(8, 8))
plt.scatter(features3[:, 0], features3[:, 1], c=lab, cmap='plasma')
for ii in np.arange(3668):
    plt.text(features3[ii,0],features3[ii,1],s=None)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-Means PCA 2D')
# plt.show()
#绘制聚类结果后3d散点图
plt.figure(figsize=(8,8))
ax = plt.subplot(111, projection='3d')
ax.scatter(features3[:, 0], features3[:, 1], features3[:, 2], c=lab, cmap='plasma')
# 视角转换，转换后更易看出簇群
ax.view_init(30, 45)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title('K-Means PCA 3D')
# plt.show()



mean_clus_feat = item_feature_data.groupby('kmeans_pre').agg('mean').reset_index()
median_clus_feat = item_feature_data.groupby('kmeans_pre').agg('median').reset_index()
mean_clus_feat.to_csv(os.path.join(current_path, 'outputs/tables/mean_clus_feat.csv'), index=None)
median_clus_feat.to_csv(os.path.join(current_path, 'outputs/tables/median_clus_feat.csv'), index=None)

# 设置半径为10，最小样本量为2，建模
dbs_feat = features3.copy()
# db = DBSCAN(eps=10, min_samples=3).fit(dbs_feat)
# labels = db.labels_
# dbs_feat['cluster_db'] = labels  # 在数据集最后一列加上经过DBSCAN聚类后的结果
# dbs_feat.sort_values('cluster_db')
y_pred = DBSCAN(eps=0.1, min_samples=10).fit_predict(dbs_feat)

# plt.figure(figsize=(8, 8))
# plt.scatter(features3[:, 0], features3[:, 1], c=y_pred, cmap='Spectral')
# for ii in np.arange(3668):
#     plt.text(features3[ii, 0], features3[ii, 1], s=None)
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('DBSCAN PCA 2D')
# plt.show()

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')
ax1.scatter(features3[:, 0], features3[:, 1], c=y_pred, cmap='Spectral')
for ii in np.arange(3668):
    ax1.text(features3[ii, 0], features3[ii, 1], s=None)
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
# ax1.set_title('DBSCAN PCA 2D')
ax2.scatter(features3[:, 0], features3[:, 1], features3[:, 2], c=y_pred, cmap='Spectral')
# 视角转换，转换后更易看出簇群
ax2.view_init(30, 45)
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_zlabel('PC3')
plt.savefig('dbscan.png')
plt.show()


