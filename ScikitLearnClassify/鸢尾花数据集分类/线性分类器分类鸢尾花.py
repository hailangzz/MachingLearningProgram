# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 13:41:23 2018

@author: Administrator
"""

from sklearn import datasets
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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
#print(x_train_std.std(),x_train_std.mean())
#print(x_test_std.std(),x_test_std.mean())

#接口训练感知器，并预测 y_pred数据···
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40,eta0=0.1,random_state=0)
ppn.fit(x_train_std,y_train)
y_pred = ppn.predict(x_test_std)

#print(y_test!=y_pred)
print('Misclassified samples: %d' % (y_test!=y_pred).sum())

#度量预测值与实际值得准确度得分，正确率···
from sklearn.metrics import accuracy_score
print('Accuracy:%.2f' % accuracy_score(y_test,y_pred))

import plot_decision_regions as pdr
import matplotlib.pyplot as plt
x_combined_std=np.vstack((x_train_std,x_test_std))
y_combined_std=np.hstack((y_train,y_test))
pdr.plot_decision_regions(X=x_combined_std,
                          y=y_combined_std,
                          classifier=ppn,
                          test_idx=range(105,105))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

#显示线性分类的错误率···
from sklearn.metrics import accuracy_score
print('Accuracy:%.2f' % accuracy_score(y_test,ppn.predict(x_test_std)))
print('Misclassified sample:%d' %(y_test!=y_pred).sum())