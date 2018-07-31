# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 09:46:08 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
variables = ['x','y','z']
labels = ['ID_0','ID_1','ID_2','ID_3','ID_4']
x = np.random.random_sample([5,3])*10
df=pd.DataFrame(x,columns=variables,index=labels)
df
from scipy.spatial.distance import pdist,squareform
row_dist=pd.DataFrame(squareform(
                      pdist(df,metric='euclidean')),
                      columns=labels,index=labels)
print(row_dist)

#初始化层次聚类的原始数据结构linkage
from scipy.cluster.hierarchy import linkage
help(linkage)
dpdist=pdist(df,metric='euclidean')
row_clusters=linkage(dpdist,method='complete')

row_clusters=pd.DataFrame(row_clusters,
             columns=['row label 1',
                      'row label 2',
                      'distance',
                      'no. of items in clust.'],
             index=['cluster %d'%(i+1) for i in range(row_clusters.shape[0])])


#绘制层次聚类图····
from scipy.cluster.hierarchy import dendrogram
row_dendr=dendrogram(row_clusters,
                     labels=labels)
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()

#凝聚聚类法聚类操作····
from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=3,
                           affinity='euclidean',
                           linkage='complete')
labels=ac.fit_predict(x)
print('Cluster labels: %s'% labels)

print(type(x),x.shape)
test_array=np.array([[7,3,2],[3,2,2],[2,3,3],[2,3,3],[3,3,3],[8,3,3]])
print(type(test_array),test_array.shape)

labels=ac.fit_predict(test_array)
print('Cluster labels: %s'% labels)