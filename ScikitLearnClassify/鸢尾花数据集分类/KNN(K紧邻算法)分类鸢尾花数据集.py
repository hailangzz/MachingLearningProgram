# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:18:42 2018

@author: Administrator
"""


from sklearn import datasets
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import plot_decision_regions as pdr

iris = datasets.load_iris()
x=iris.data[:,[2,3]]
y=iris.target

#按照比例划分训练数据集及测试数据集
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

#对数据记录的特征字段进行标准化处理(转化为标准正态分布)···
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train_std=sc.transform(x_train)
x_test_std=sc.transform(x_test)

#K近邻算法对记录字段数据的数量级很敏感因此需要标准化处理····
#K紧邻算法实现······
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,p=2,
                           metric='minkowski')
knn.fit(x_train_std,y_train)
pdr.plot_decision_regions(x_train_std,
                          y_train,
                          classifier=knn,
                          test_idx=range(105,105))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

from sklearn.metrics import accuracy_score
print(' k近邻算法分类')
print('Accuracy:%.2f' % accuracy_score(y_test,knn.predict(x_test_std)))
print('Misclassified sample:%d' %(y_test!=knn.predict(x_test_std)).sum())
