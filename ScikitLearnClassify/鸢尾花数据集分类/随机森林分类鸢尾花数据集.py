# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:01:48 2018

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


#随机森林分类方法实现····
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                random_state=1,
                                n_jobs=2)

forest.fit(x_train,y_train)
pdr.plot_decision_regions(x_train,
                          y_train,
                          classifier=forest,
                          test_idx=range(105,105))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

#显示决策树分类的错误率···
from sklearn.metrics import accuracy_score
print('Accuracy:%.2f' % accuracy_score(y_test,forest.predict(x_test)))
print('Misclassified sample:%d' %(y_test!=forest.predict(x_test)).sum())