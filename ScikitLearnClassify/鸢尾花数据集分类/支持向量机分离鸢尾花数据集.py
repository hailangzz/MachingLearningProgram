# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:00:46 2018

@author: Administrator
"""

from sklearn import datasets
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

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
x_combined_std=np.vstack((x_train_std,x_test_std))
y_combined_std=np.hstack((y_train,y_test))

from sklearn.svm import SVC
import plot_decision_regions as pdr
svm = SVC(kernel='rbf',random_state=0,gamma=0.2,C=1.0)
svm.fit(x_train_std,y_train)
pdr.plot_decision_regions(x_combined_std,
                          y_combined_std,
                          classifier=svm,
                          test_idx=range(105,105))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

from sklearn.metrics import accuracy_score
print('gamma values is :0.2')
print('Accuracy:%.2f' % accuracy_score(y_test,svm.predict(x_test_std)))
print('Misclassified sample:%d' %(y_test!=svm.predict(x_test_std)).sum())



#调整gamma参数大小，所得到的支持向量机分类模型·····
svm = SVC(kernel='rbf',random_state=0,gamma=10,C=1.0)
svm.fit(x_train_std,y_train)
pdr.plot_decision_regions(x_combined_std,
                          y_combined_std,
                          classifier=svm,
                          test_idx=range(105,105))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

from sklearn.metrics import accuracy_score
print('gamma values is :10')
print('Accuracy:%.2f' % accuracy_score(y_test,svm.predict(x_test_std)))
print('Misclassified sample:%d' %(y_test!=svm.predict(x_test_std)).sum())

#调整gamma参数大小，所得到的支持向量机分类模型·····
svm = SVC(kernel='rbf',random_state=0,gamma=0.01,C=1.0)
svm.fit(x_train_std,y_train)
pdr.plot_decision_regions(x_combined_std,
                          y_combined_std,
                          classifier=svm,
                          test_idx=range(105,105))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

from sklearn.metrics import accuracy_score
print('gamma values is :0.01')
print('Accuracy:%.2f' % accuracy_score(y_test,svm.predict(x_test_std)))
print('Misclassified sample:%d' %(y_test!=svm.predict(x_test_std)).sum())