# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:33:13 2018

@author: Administrator
"""
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from pylab import *

#mpl.rcParams['font.sans-serif'] = ['SimHei']
#mpl.rcParams['axes.unicode_minus'] = False

def lin_regplot(x,y,model):
    plt.scatter(x,y,c='blue')
    plt.plot(x,model.predict(x),color='red')
    return None


dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                        header=None,sep='\s+')

dataframe.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
x=dataframe[['RM']].values
y=dataframe[['MEDV']].values

slr=LinearRegression()
slr.fit(x,y)
print(slr.coef_[0],slr.intercept_)

lin_regplot(x,y,slr)
plt.xlabel(u'平均房间数量',)
plt.ylabel(u'每平米房屋均价',)
plt.show()

