# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 18:02:46 2018

@author: Administrator
"""


from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge,Lasso,ElasticNet



def lin_regplot(x,y,model):
    plt.scatter(x,y,c='blue')
    plt.plot(x,model.predict(x),color='red')
    return None


dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                        header=None,sep='\s+')

dataframe.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
x=dataframe.iloc[:,:-1].values
y=dataframe[['MEDV']].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
#y_train=y_train[0]
#x_train=x_train[0]


reg = Ridge(alpha=0.5)
reg.fit(x,y)
y_train_pred = reg.predict(x_train)
y_test_pred = reg.predict(x_test)
y_train_pred=y_train_pred.reshape(y_train_pred.shape[0],1)
y_test_pred=y_test_pred.reshape(y_test_pred.shape[0],1)

#绘制多元回归的残差图像·····
plt.scatter(y_train_pred,y_train_pred-y_train,c='blue',marker='o',label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='lightgreen',marker='s',label='test data')

plt.xlabel('predicted values')
plt.ylabel('residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color='red')
plt.xlim([-10,50])
plt.show()
print('系数项：',reg.coef_,'\n')
print('截距项：',reg.intercept_,'\n')
print("训练集均方误差",mean_squared_error(y_train,y_train_pred))
print('测试集均方误差',mean_squared_error(y_test,y_test_pred))

print("训练集决定系数",r2_score(y_train,y_train_pred))
print("测试集决定系数",r2_score(y_test,y_test_pred))

reg = Ridge(alpha=0.01)
reg.fit(x,y)
y_train_pred = reg.predict(x_train)
y_test_pred = reg.predict(x_test)
y_train_pred=y_train_pred.reshape(y_train_pred.shape[0],1)
y_test_pred=y_test_pred.reshape(y_test_pred.shape[0],1)

#绘制多元回归的残差图像·····
plt.scatter(y_train_pred,y_train_pred-y_train,c='blue',marker='o',label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='lightgreen',marker='s',label='test data')

plt.xlabel('predicted values')
plt.ylabel('residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color='red')
plt.xlim([-10,50])
plt.show()
print('系数项：',reg.coef_,'\n')
print('截距项：',reg.intercept_,'\n')
print("训练集均方误差",mean_squared_error(y_train,y_train_pred))
print('测试集均方误差',mean_squared_error(y_test,y_test_pred))

print("训练集决定系数",r2_score(y_train,y_train_pred))
print("测试集决定系数",r2_score(y_test,y_test_pred))

lass = Lasso(alpha=0.5)
lass.fit(x,y)
y_train_pred = lass.predict(x_train)
y_test_pred = lass.predict(x_test)
y_train_pred=y_train_pred.reshape(y_train_pred.shape[0],1)
y_test_pred=y_test_pred.reshape(y_test_pred.shape[0],1)

#绘制多元回归的残差图像·····
plt.scatter(y_train_pred,y_train_pred-y_train,c='blue',marker='o',label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='lightgreen',marker='s',label='test data')

plt.xlabel('predicted values')
plt.ylabel('residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color='red')
plt.xlim([-10,50])
plt.show()
print('系数项：',lass.coef_,'\n')
print('截距项：',lass.intercept_,'\n')
print("训练集均方误差",mean_squared_error(y_train,y_train_pred))
print('测试集均方误差',mean_squared_error(y_test,y_test_pred))

print("训练集决定系数",r2_score(y_train,y_train_pred))
print("测试集决定系数",r2_score(y_test,y_test_pred))


elast = ElasticNet(alpha=0.5,l1_ratio=0.5)
elast.fit(x,y)
y_train_pred = elast.predict(x_train)
y_test_pred = elast.predict(x_test)
y_train_pred=y_train_pred.reshape(y_train_pred.shape[0],1)
y_test_pred=y_test_pred.reshape(y_test_pred.shape[0],1)

#绘制多元回归的残差图像·····
plt.scatter(y_train_pred,y_train_pred-y_train,c='blue',marker='o',label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='lightgreen',marker='s',label='test data')

plt.xlabel('predicted values')
plt.ylabel('residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color='red')
plt.xlim([-10,50])
plt.show()
print('系数项：',elast.coef_,'\n')
print('截距项：',elast.intercept_,'\n')
print("训练集均方误差",mean_squared_error(y_train,y_train_pred))
print('测试集均方误差',mean_squared_error(y_test,y_test_pred))
print("训练集均方误差",mean_squared_error(y_train,y_train_pred))
print('测试集均方误差',mean_squared_error(y_test,y_test_pred))

print("训练集决定系数",r2_score(y_train,y_train_pred))
print("测试集决定系数",r2_score(y_test,y_test_pred))

