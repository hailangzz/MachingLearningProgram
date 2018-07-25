# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:28:11 2018

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

slr = LinearRegression()
slr.fit(x_train,y_train)
y_train_pred = slr.predict(x_train)
y_test_pred = slr.predict(x_test)


#绘制多元回归的残差图像·····
plt.scatter(y_train_pred,y_train_pred-y_train,c='blue',marker='o',label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='lightgreen',marker='s',label='test data')

plt.xlabel('predicted values')
plt.ylabel('residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color='red')
plt.xlim([-10,50])
plt.show()

print("训练集残差",mean_squared_error(y_train,y_train_pred))
print('测试集残差',mean_squared_error(y_test,y_test_pred))

print("训练集决定系数",r2_score(y_train,y_train_pred))
print("测试集决定系数",r2_score(y_test,y_test_pred))
