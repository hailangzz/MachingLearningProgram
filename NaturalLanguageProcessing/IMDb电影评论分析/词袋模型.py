# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 16:04:32 2018

@author: Administrator
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

#将评论语句转化为词袋向量····
count = CountVectorizer()
docs=np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining and the weather is sweet',])
bag = count.fit_transform(docs)
WordDict=count.vocabulary_
WordVect=bag.toarray()
#print(WordDict)
#print(WordVect)

#削弱常见词在词袋向量生成时，对词频的统计影响。使用逆文档频率统计···
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
tfidf_WordVect = tfidf.fit_transform(WordVect)
print(tfidf_WordVect)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())