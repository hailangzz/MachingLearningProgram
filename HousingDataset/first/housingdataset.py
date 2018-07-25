# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 10:22:30 2018

@author: Administrator
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                        header=None,sep='\s+')

dataframe.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
#print (dataframe.columns)
#print(dataframe.head())

sns.set(style='whitegrid',context='notebook')
columns=['LSTAT','INDUS','NOX','RM','MEDV']
sns.pairplot(dataframe[columns],size=2.5)
#plt.show()

cm = np.corrcoef(dataframe[columns].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':15},yticklabels=columns,xticklabels=columns)
#plt.show()


class LinearRegressionGD(object):
    def __init__(self,eta=0.001,n_iter=20):
        self.eta=eta;
        self.n_iter=n_iter
        
    def fit(self,x,y):
        self.w_=np.zeros(1+x.shape[1])
        self.cost_=[]
        
        for i in range(self.n_iter):
            output = self.net_input(x)
            errors=(y-output)
            self.w_[1:]+=self.eta * x.T.dot(errors)
            self.w_[0]+=self.eta * errors.sum()
            cost=(errors**2).sum()/2.0
            self.cost_.append(cost)
        return self
    def net_input(self,x):
        return np.dot(x,self.w_[1:]+self.w_[0])
    
    def predict(self,x):
        return self.net_input(x)
    
x=dataframe[['RM']].values
y=dataframe[['MEDV']].values

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x_std=sc_x.fit_transform(x)
y_std=sc_y.fit_transform(y)
#print(x[0:10])
#print(x_std[0:10])
#print(x_std)
#a=np.sqrt(np.sum(np.square(x_std)))
#print(a)
#print(x_std.sum()/len(x_std))
#print(np.std(x_std))
lr=LinearRegressionGD()
lr.fit(x_std,y_std)

plt.plot(range(1,lr.n_iter+1),lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
#plt.show()

