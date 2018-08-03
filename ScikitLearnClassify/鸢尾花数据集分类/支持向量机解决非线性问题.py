# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 08:26:37 2018

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt

#生成亦、或原始数据集及数据图像····
np.random.seed(0)
x_xor = np.random.randn(2000,2)
y_xor = np.logical_xor(x_xor[:,0]>0,x_xor[:,1]>0)
y_xor = np.where(y_xor,1,-1)

plt.scatter(x_xor[y_xor==1,0],x_xor[y_xor==1,1],
            c='b',marker='x',label='1')
plt.scatter(x_xor[y_xor==-1,0],x_xor[y_xor==-1,1],
            c='r',marker='s',label='-1')

plt.ylim(-3.0)
plt.legend()
plt.show()

#svm核函数非线性拟合，创建分离超平面····
from sklearn.svm import SVC
import plot_decision_regions as pdr
svm_hook = SVC(kernel='rbf',random_state=0,gamma=0.1,C=10.0)
svm_hook.fit(x_xor,y_xor)
pdr.plot_decision_regions(x_xor,y_xor,classifier=svm_hook)
plt.legend(loc='upper left')
plt.show()

from sklearn.metrics import accuracy_score
print('svm核函数非线性拟合分类结果评估：')
print('Accuracy:%.2f' % accuracy_score(y_xor,svm_hook.predict(x_xor)))
print('Misclassified sample:%d' %(y_xor!=svm_hook.predict(x_xor)).sum())
print(svm_hook.predict([[1,-1],[5,5]]))
########################################################################
#svm核函数线性拟合，创建分离超平面····
from sklearn.svm import SVC
import plot_decision_regions as pdr
svm_linear = SVC(kernel='linear',random_state=0,gamma=0.1,C=10.0)
svm_linear.fit(x_xor,y_xor)
pdr.plot_decision_regions(x_xor,y_xor,classifier=svm_linear)
plt.legend(loc='upper left')
plt.show()

from sklearn.metrics import accuracy_score
print('svm核函数线性拟合分类结果评估：')
print('Accuracy:%.2f' % accuracy_score(y_xor,svm_linear.predict(x_xor)))
print('Misclassified sample:%d' %(y_xor!=svm_linear.predict(x_xor)).sum())
print(svm_hook.predict([[1,-1],[5,5]]))


##########################################################################
#逻辑斯谛回归分类及错误率分析····
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0,random_state=0)
lr_modle=lr.fit(x_xor,y_xor)
test_idx=range(105,105)
pdr.plot_decision_regions(x_xor,
                          y_xor,
                          classifier=lr,
                          test_idx=range(105,105))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

from sklearn.metrics import accuracy_score
print('逻辑斯谛回归分类结果评估：')
print('Accuracy:%.2f' % accuracy_score(y_xor,lr.predict(x_xor)))
print('Misclassified sample:%d' %(y_xor!=lr.predict(x_xor)).sum())

#########################################################################
#线性分类器分类方法····
from  sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40,eta0=0.1,random_state=0)
ppn.fit(x_xor,y_xor)
y_pred = ppn.predict(x_xor)

pdr.plot_decision_regions(x_xor,
                          y_xor,
                          classifier=lr,
                          test_idx=range(105,105))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

from sklearn.metrics import accuracy_score
print('线性分类法分类结果评估：')
print('Accuracy:%.2f' % accuracy_score(y_xor,y_pred))
print('Misclassified sample:%d' %(y_xor!=y_pred).sum())

###########################################################################
#决策树分类器分类方法···
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=3,random_state=0)
tree.fit(x_xor,y_xor)

test_idx=range(105,105)
pdr.plot_decision_regions(x_xor,
                          y_xor,
                          classifier=tree,
                          test_idx=range(105,105))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

#显示决策树分类的错误率···
from sklearn.metrics import accuracy_score
print('决策树分类结果评估：')
print('Accuracy:%.2f' % accuracy_score(y_xor,tree.predict(x_xor)))
print('Misclassified sample:%d' %(y_xor!=tree.predict(x_xor)).sum())