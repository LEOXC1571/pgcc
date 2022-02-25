# Coded By LEO XU
# At 2022/2/25 19:03

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

current_path = os.path.dirname(os.path.realpath(__file__))
item_feature_data = pd.read_csv(os.path.join(current_path, 'raw_datasets/item_feature_data.csv'), header=0)
feature_col = item_feature_data.columns.tolist()[1:18]
feat_corr = item_feature_data[feature_col].corr()

'''mask=np.zeros_like(feat_corr)
mask[np.triu_indices_from(mask)]=True
plt.figure(figsize=(15,15))
with sns.axes_style("white"):
    ax=sns.heatmap(feat_corr,mask=mask,square=True,annot=True,cmap='bwr')
ax.set_title("feat_corr Variables Relation")
plt.show()'''

'''sns.set_style("dark")
plt.figure(figsize = (10,10))
# np.zero_like的意思就是生成一个和你所给数组a相同shape的全0数组。
mask = np.zeros_like(feat_corr, dtype=bool)
# np.triu_indices_from()返回方阵的上三角矩阵的索引
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
ax = sns.heatmap(feat_corr, mask=mask, cmap=cmap, square=True, annot=True,fmt='0.2f')
ax.set_title("feat_corr Variables Relation")
plt.show()'''

##肘方法看k值，簇内离差平方和
# 对每一个k值进行聚类并且记下对于的SSE，然后画出k和SSE的关系图


sse = []
for i in range(1, 15):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    km.fit(item_feature_data[feature_col])
    sse.append(km.inertia_)

plt.plot(range(1, 15), sse, marker='*')
plt.xlabel('n_clusters')
plt.ylabel('distortions')
plt.title("The Elbow Method")
plt.show()

# 进行K-Means聚类分析
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
kmeans.fit(item_feature_data[feature_col])
lab = kmeans.predict(item_feature_data[feature_col])
lab = lab.T
lab_result = pd.DataFrame(lab)
item_feature_data = pd.concat([item_feature_data, lab_result], ignore_index=False, axis=1)
item_feature_data.rename(columns={'0': 'kmeans_pre'}, inplace=True)
item_feature_data.to_csv(os.path.join(current_path, 'raw_datasets/kmeans_pre.csv'),index=None)
# 绘制聚类结果2维的散点图
'''plt.figure(figsize=(8,8))
plt.scatter(item_feature_data[feature_col][:,0],item_feature_data[feature_col][:,1],c=lab)
for ii in np.arange(205):
    plt.text(item_feature_data[feature_col][ii,0],item_feature_data[feature_col][ii,1],s=car_price.car_ID[ii])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('K-Means PCA')
plt.show()'''

# 绘制聚类结果后3d散点图

'''plt.figure(figsize=(8,8))
ax=plt.subplot(111,projection='3d')
ax.scatter(item_feature_data[feature_col][:,0],item_feature_data[feature_col][:,1],item_feature_data[feature_col][:,2],c=lab)
#视角转换，转换后更易看出簇群
ax.view_init(30,45)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()'''
