# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:10:52 2018

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

BillData = pd.read_table(r'E:\GitLocalRepository\MachingLearningProgram\0807\ALLResult0807\Result_0807_1',
                         header=None,encoding='gb2312',delim_whitespace=True)
BillData.columns=['PhoneNumber', 'ARPU', 'DurationDial']

TravelData = pd.read_table(r'E:\GitLocalRepository\MachingLearningProgram\0807\ALLResult0807\Result_0807_3',
                         header=None,encoding='gb2312',delim_whitespace=True)
TravelData.columns=['PhoneNumber', 'TravelNumber', 'FeeOfTravel']

HotelData = pd.read_table(r'E:\GitLocalRepository\MachingLearningProgram\0807\ALLResult0807\Result_0807_4',
                         header=None,encoding='gb2312',delim_whitespace=True)
HotelData.columns=['PhoneNumber', 'HotelNumber', 'FeeOfHotel']

CreditcardBillData = pd.read_table(r'E:\GitLocalRepository\MachingLearningProgram\0807\ALLResult0807\Result_0807_4',
                         header=None,encoding='gb2312',delim_whitespace=True)
HotelData.columns=['PhoneNumber', 'HotelNumber', 'FeeOfHotel']

PhoneNumber=BillData.iloc[:,0]
ARPU=BillData.iloc[:,1]
time=BillData.iloc[:,2]
plt.scatter(x=time,y=ARPU,marker='.',linewidths=1)
plt.xlim(0,4100)
plt.show()

print(BillData[BillData.iloc[:,0]==13509691147])

x_train,x_test=train_test_split(BillData,test_size=0.1,random_state=0)
x_test['PhoneNumber'].to_csv(r'f:/0807.csv',index=None)
