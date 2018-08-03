# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:26:23 2018

@author: Administrator
"""

import pyprind
import pandas as pd
import os
import numpy as np

#pandas 读取Excel中的csv文件···
df = pd.read_csv('E:/GitLocalRepository/movie_data.csv')


#清洗文本数据····
#print(df.loc[0,'review'],df.loc[0,'sentiment'])
print(df.loc[0,'review'][-50:])

#使用python正则表达式清洗字符串··· import re
import re

def preprocessor(test):
    
    
    