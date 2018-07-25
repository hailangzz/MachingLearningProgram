# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:58:09 2018

@author: Administrator
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import RANSACRegressor


def lin_regplot(x,y,model):
    plt.scatter(x,y,c='blue')
    plt.plot(x,model.predict(x),color='red')
    return None


dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                        header=None,sep='\s+')

dataframe.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
x=dataframe[['RM']].values
y=dataframe[['MEDV']].values


ransac = RANSACRegressor(
                         LinearRegression(),
                         max_trials=100,
                         min_samples=50,
                         residual_metric=lambda x: np.sum(np.abs(x),axis=1),
                         residual_threshold=5.0,
                         random_state=0)

ransac.fit(x,y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_x = np.arange(3,10,1)
line_y_ransac = ransac.predict(line_x[:,np.newaxis])

plt.scatter(x[inlier_mask],y[inlier_mask],c='blue',marker='o',label='Inliers')
plt.scatter(x[outlier_mask],y[outlier_mask],c='lightgreen',marker='s',label='Outliers')
plt.plot(line_x,line_y_ransac,color='red')

plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $ 1000\'s [MEdv]')
plt.legend(loc='upper left')
plt.show()
print(ransac.estimator_.coef_[0])
print(ransac.estimator_.intercept_)


