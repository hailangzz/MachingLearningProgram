# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:17:31 2018

@author: Administrator
"""
import re
span1=re.match('www','www.runoob.com').span()
match1=re.match('www','www.runoob.com')
match2=re.match('www','www.runoob.com')
print(re.match('www','www.runoob.com').span())
print(re.match('www','www.runoob.com'))
print(re.match('com','www.runoob.com'))