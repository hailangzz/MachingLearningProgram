# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 16:46:48 2018

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


#逻辑斯谛回归分类及错误率分析····
from sklearn.linear_model import LogisticRegression
import plot_decision_regions as pdr
import matplotlib.pyplot as plt
x_combined_std=np.vstack((x_train_std,x_test_std))
y_combined_std=np.hstack((y_train,y_test))

lr = LogisticRegression(C=1000.0,random_state=0)
lr_modle=lr.fit(x_train_std,y_train)
print(lr.coef_[0],lr.coef_[1],lr.coef_[2])
print(lr_modle)
test_idx=range(105,105)
pdr.plot_decision_regions(x_combined_std,
                          y_combined_std,
                          classifier=lr,
                          test_idx=range(105,105))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()


from sklearn.metrics import accuracy_score
print('Accuracy:%.2f' % accuracy_score(y_test,lr.predict(x_test_std)))
print('Misclassified sample:%d' %(y_test!=lr.predict(x_test_std)).sum())