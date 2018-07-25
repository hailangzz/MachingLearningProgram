# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:32:37 2018

@author: Administrator
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                        header=None,sep='\s+')

dataframe.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

x = dataframe[['LSTAT']].values
y = dataframe[['MEDV']].values
regr = LinearRegression()

quadratic = PolynomialFeatures(degree = 2)
cubic = PolynomialFeatures(degree = 3)
x_quad = quadratic.fit_transform(x)
x_cubic = cubic.fit_transform(x)

x_fit = np.arange(x.min(),x.max(),1)[:,np.newaxis]
#x_fit=x_fit[:,np.newaxis]
regr = regr.fit(x,y)
y_lin_fit = regr.predict(x_fit)
linear_r2 = r2_score(y,regr.predict(x))


regr = regr.fit(x_quad,y)
y_quad_fit = regr.predict(quadratic.fit_transform(x_fit))
quadratic_r2 = r2_score(y,regr.predict(x_quad))

regr = regr.fit(x_cubic,y)
y_cubic_fit = regr.predict(cubic.fit_transform(x_fit))
cubic_r2 = r2_score(y,regr.predict(x_cubic))


plt.scatter(x,y,label='training points', color='lightgray')
plt.plot(x_fit,y_lin_fit,label='linear (d=1) , $R^1=%.2f$'% linear_r2,color='blue',lw=2,linestyle=':')
plt.plot(x_fit,y_quad_fit,label='linear (d=2) , $R^2=%.2f$'% quadratic_r2,color='red',lw=2,linestyle='-')
plt.plot(x_fit,y_cubic_fit,label='linear (d=3) , $R^3=%.2f$'% cubic_r2,color='green',lw=2,linestyle='--')

plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('price in $1000\'s [MEDV]')
plt.legend(loc='uppper right')
plt.show()


# log对数模型建模····
x_log = np.log(x)
y_sqrt = np.sqrt(y)

x_fit = np.arange(x_log.min()-1,x_log.max()+1,1)[:,np.newaxis]
regr = regr.fit(x_log,y_sqrt)
y_lin_fit = regr.predict(x_fit)
y_lin_fit1=regr.predict(x_log)
linear_r2 = r2_score(y_sqrt, regr.predict(x_log))

plt.scatter(x_log,y_sqrt,label='trating points',color='lightgray')
plt.plot(x_fit,y_lin_fit,label='linear (d=1) $R^2=%.2f$'% linear_r2, color='blue',lw=2)
plt.plot(x_log,y_lin_fit1,label='linear (d=1) $R^2=%.2f$'% linear_r2, color='red',lw=2)

plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \;in\;\$1000\'s[MEDV]}$')
plt.legend(loc='lower left')
plt.show()

