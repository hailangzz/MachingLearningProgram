# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 14:36:59 2018

@author: Administrator
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split

dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                        header=None,sep='\s+')

dataframe.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

x = dataframe.iloc[:,:-1].values
y = dataframe[['MEDV']].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

forest = RandomForestRegressor(n_estimators=1000,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1)

forest.fit(x_train,y_train)
y_train_pred = forest.predict(x_train)[:,np.newaxis]
y_test_pred = forest.predict(x_test)[:,np.newaxis]

print('训练集的均方误差：',mean_squared_error(y_train,y_train_pred))
print('测试集的均方误差：',mean_squared_error(y_test,y_test_pred))

print('训练集的决定系数：',r2_score(y_train,y_train_pred))
print('测试集的决定系数：',r2_score(y_test,y_test_pred))

plt.scatter(y_train_pred,
            y_train_pred-y_train,
            color='black',
            marker='o',
            s=20,
            alpha=0.5,
            label='Training data')
plt.scatter(y_test_pred,
            y_test_pred-y_test,
            c='lightgreen',
            marker='s',
            s=35,
            alpha=0.7,
            label='Test data',
            )
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color='red')
plt.xlim([-10,50])
plt.show()