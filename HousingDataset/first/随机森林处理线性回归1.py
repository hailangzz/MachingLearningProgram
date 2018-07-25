# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:55:26 2018

@author: Administrator
"""


from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                        header=None,sep='\s+')

dataframe.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

x = dataframe[['LSTAT']].values
y = dataframe[['MEDV']].values

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(x,y)
sort_idx = x.flatten().argsort()
lin_regplot(x[sort_idx],y[sort_idx],tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.show()
print(tree.predict([[1],[2],[3],[4],[5]]))
