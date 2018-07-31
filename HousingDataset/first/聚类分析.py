# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 15:25:33 2018

@author: Administrator
"""

from sklearn.datasets import make_blobs

x,y = make_blobs(n_samples=150,
                 n_features=2,
                 centers=3,
                 cluster_std=0.5,
                 shuffle=True,
                 random_state=0)

import matplotlib.pyplot as plt
plt.scatter(x[:,0],
            x[:,1],
            c='green',
            marker='o',
            s=50)

plt.grid()
plt.show()       

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(x) 

print(x[:,1].std())

plt.scatter(x[y_km==0,0],
            x[y_km==0,1],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1')


plt.scatter(x[y_km==1,0],
            x[y_km==1,1],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2')


plt.scatter(x[y_km==2,0],
            x[y_km==2,1],
            s=50,
            c='lightblue',
            marker='v',
            label='cluster 3')

plt.scatter(km.cluster_centers_[:,0],
            km.cluster_centers_[:,1],
            s=250,
            c='red',
            marker='*',
            label='centroids')
print(y_km==0)
flag=y_km==0
print(type(flag),type(y_km))
cluster_centers_=km.cluster_centers_
plt.legend()
plt.grid()
plt.show()

#输出族内误差平方和···
print('Distortion: %.2f'% km.inertia_)
distortions=[]
for i in range(1,11):
    km = KMeans(n_clusters=i,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
    km.fit(x)
    distortions.append(km.inertia_)
plt.plot(range(1,11),distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.grid()
plt.show()
    