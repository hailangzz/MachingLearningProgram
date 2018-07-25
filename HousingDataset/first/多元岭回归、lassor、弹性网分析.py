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
from sklearn import linear_model
from sklearn.linear_model import Lasso,ElasticNet


def lin_regplot(x,y,model):
    plt.scatter(x,y,c='blue')
    plt.plot(x,model.predict(x),color='red')
    return None


dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                        header=None,sep='\s+')

dataframe.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
x=dataframe.iloc[:,:-1].values
y=dataframe[['MEDV']].values
reg = linear_model.Ridge(alpha=0.5)
reg.fit(x,y)
print('系数项：',reg.coef_,'\n')
print('截距项：',reg.intercept_,'\n')

lass = Lasso(alpha=0.5)
lass.fit(x,y)
print('系数项：',lass.coef_,'\n')
print('截距项：',lass.intercept_,'\n')

elast = ElasticNet(alpha=0.5,l1_ratio=0.5)
elast.fit(x,y)
print('系数项：',elast.coef_,'\n')
print('截距项：',elast.intercept_,'\n')

