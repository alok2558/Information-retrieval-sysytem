#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import os
import glob
import nltk
from nltk.collocations import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
Stopwords = set(stopwords.words('english'))
import pickle
from nltk.stem import PorterStemmer
from collections import Counter
import re
ps = PorterStemmer()
import sys


# In[15]:


Stopwords = set(stopwords.words('english'))
import re


# In[22]:


with open(sys.argv[1],'r') as file:
    q_file = file.read()
    file.close()
    
q_dict = {}
for l in q_file.split('\n'):
    if(len(l.split('\t'))==2):
        
        q_id,query = l.split('\t')
        
        q_dict[q_id] = query


# In[23]:


q_dict


# In[24]:


with open('21111008-ir-systems/post_file.pkl', 'rb') as f:
    tf = pickle.load(f)
    f.close()

with open('21111008-ir-systems/file_num.pkl', 'rb') as f:
    file_num = pickle.load(f)
    f.close()
with open('21111008-ir-systems/df.pkl', 'rb') as f:
    df = pickle.load(f)
    f.close()
with open('21111008-ir-systems/length.pkl', 'rb') as f:
    length = pickle.load(f)
    f.close()
with open('21111008-ir-systems/l2_norm.pkl', 'rb') as f:
    l2_norm = pickle.load(f)
    f.close()


# Boolean

# In[25]:



def BooleanRetreval(qry): 
    q_list=[]
    regex = re.compile('[^a-zA-Z0-9\s]')
    inputt = re.sub(regex,'',qry)
    inputt = re.sub(re.compile('\d'),'',inputt)
    inputt = word_tokenize(inputt)
    for i in range(len(inputt)):
        inputt[i] = ps.stem(inputt[i])
    inputt = [i for i in inputt if i not in Stopwords]

    ls=[]
    for i in range(8635):
        ls.append(0)
    for j in tf[inputt[0]].keys():
        ls[j] = 1
    if(len(inputt) >1):
        for i in range(1,len(inputt)):
            for j in tf[inputt[i]].keys():
                if(ls[j] ==i):
                    ls[j] = ls[j] + 1
    for i in range(8635):
        if(ls[i] == len(inputt)):
            q_list.append(file_num[i])
    return q_list


# smb
# 

# In[31]:


def tf_idf(qry):
    q_list=[]
    import math
    regex = re.compile('[^a-zA-Z0-9\s]')
    inputt = re.sub(regex,'',qry)
    inputt = re.sub(re.compile('\d'),'',inputt)
    inputt = word_tokenize(inputt)
    for i in range(len(inputt)):
        inputt[i] = ps.stem(inputt[i])
    inputt = [i for i in inputt if i not in Stopwords]
    lis = []

    for j in range(8635):
        temp = []
        for i in inputt:
            try:
                tf_idf = tf[i][j]*math.log(8635/df[i])
            except KeyError:
                tf_idf = 0
            tf_idf = tf_idf/l2_norm[j]
            temp.append(tf_idf)
        lis.append(temp)
    norm = 0
    temp_tf_idf =[]

    for i in inputt:
        qtf_idf = inputt.count(i)*math.log(8635/df[i])
        norm = norm + qtf_idf**2
        norm = math.sqrt(norm)
        temp_tf_idf.append(qtf_idf)

    for i in range(len(temp_tf_idf)):
        temp_tf_idf[i] = temp_tf_idf[i]/norm

    top=[]
    for i in range(8635):
        top.append(np.dot(lis[i],temp_tf_idf))
    for i in range(30):
        max_value = max(top)
        q_list.append(file_num[top.index(max_value)])
        top[top.index(max_value)] = 0
    return q_list


# bm 25

# In[35]:


def bm25(qry):
    q_list=[]
    k=1.2
    b = 0.75
    summ = 0
    for i in range(8635):
        summ = summ + length[i]
    l_avg = summ/8635
    import math
    regex = re.compile('[^a-zA-Z0-9\s]')
    inputt = re.sub(regex,'',qry)
    inputt = re.sub(re.compile('\d'),'',inputt)
    inputt = word_tokenize(inputt)
    for i in range(len(inputt)):
        inputt[i] = ps.stem(inputt[i])
    inputt = [i for i in inputt if i not in Stopwords]
    ls = []
    for i in range(8635):
        x =0
        for j in inputt:
            try:
                idf = math.log((8635-df[j]+0.5)/(df[j]+0.5))
            except KeyError:
                idf =0
            try:
                tff= tf[j][i]
            except KeyError:
                tff = 0
            x = x + idf*(k+1)*tff/(tff+k*(1-b+b*(length[i]/l_avg)))
        ls.append(x)
        
    for i in range(15):
        max_value = max(ls)
        q_list.append(file_num[ls.index(max_value)])
        ls[ls.index(max_value)] = 0
    return q_list


# In[36]:


smb = 'QueryId,Iteration,DocId,Relevance'
tfidf = 'QueryId,Iteration,DocId,Relevance'
bmtf = 'QueryId,Iteration,DocId,Relevance'
smb = 'QueryId,Iteration,DocId,Relevance'

for q_id in q_dict:
    qry = q_dict[q_id]
    
    q_list = BooleanRetreval(qry)
    c = 0
    for i in q_list:
        if c == 10:break
        c += 1
        smb = smb + str('\n'+q_id+','+'1,'+i+','+'1')
    
    k = 0
    while (c<10):
        if file_num[k] not in q_list:
            smb = smb + str('\n'+q_id+','+'1,'+file_num[k]+','+'0')
            c += 1
        else:
            k += 1
    
    q_list = tf_idf(qry)
    c = 0
    for i in q_list:
        if c == 10:break
        c += 1
        tfidf = tfidf + str('\n'+q_id+','+'1,'+i+','+'1')
    
    k = 0
    while (c<10):
        if file_num[k] not in q_list:
            tfidf = tfidf + str('\n'+q_id+','+'1,'+file_num[k]+','+'0')
            c += 1
        else:
            k += 1 
    
    
    q_list = bm25(qry)
    c = 0
    for i in q_list:
        if c == 10:break
        c += 1
        bmtf = bmtf + str('\n'+q_id+','+'1,'+i+','+'1')
    
    k = 0
    while (c<10):
        if file_num[k] not in q_list:
            bmtf = bmtf + str('\n'+q_id+','+'1,'+file_num[k]+','+'0')
            c += 1
        else:
            k += 1


# In[39]:


with open('Output/Boolean.csv','w') as file:
    file.write(smb)
    file.close()
with open('Output/Tf-Idf.csv','w') as file:
    file.write(tfidf)
    file.close()
with open('Output/BM25.csv','w') as file:
    file.write(bmtf)
    file.close()

