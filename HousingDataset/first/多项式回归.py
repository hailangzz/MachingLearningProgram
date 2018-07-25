# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 09:19:09 2018

@author: Administrator
"""

from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

x=np.array([258.0,270.0,294.0,320.0,342.0,368.0,396.0,446.0,480.0,586.0,])[:,np.newaxis]
y=np.array([236.4,234.4,252.8,298.6,314.2,342.2,360.8,368.0,391.2,390.8])

lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree = 2)
x_quad = quadratic.fit_transform(x)

x_fit = np.arange(250,600,10)[:,np.newaxis]
lr.fit(x,y)
y_lin_fit = lr.predict(x_fit)[:,np.newaxis]


pr.fit(x_quad,y)
y_quad_fit1 = pr.predict(x_quad)
y_quad_fit = pr.predict(quadratic.fit_transform(x_fit))[:,np.newaxis]


plt.scatter(x,y,label='traing points')
plt.plot(x_fit,y_lin_fit,label='linear fit',linestyle='--')
plt.plot(x_fit,y_quad_fit,label='quadratic fit')
plt.plot(x,y_quad_fit1,label='quadratic fit1')

plt.legend(loc='upper left')
plt.show()

y_lin_pred = lr.predict(x)
y_quad_pred = pr.predict(x_quad)
print('一元线性拟合均方误差：',mean_squared_error(y,y_lin_pred))
print('一元线非线性拟合均方误差：',mean_squared_error(y,y_quad_pred))

print('一元线性拟合判定系数：',r2_score(y,y_lin_pred))
print('一元非线性拟合判定系数：',r2_score(y,y_quad_pred))