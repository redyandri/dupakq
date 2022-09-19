#!/usr/bin/env python
# coding: utf-8

# In[17]:


from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory,StopWordRemover,ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
import numpy as np #Operasi Matematika dan linear aljebra 
import pandas as pd #data processing
import matplotlib.pyplot as plt #Visualisasi data 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns #Visualisasi data import string
import nltk
# nltk.download("punkt")
from nltk.tokenize import RegexpTokenizer 
# from nltk.corpus import stopwords nltk.download('stopwords')
from nltk.stem import PorterStemmer
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import TfidfVectorizer 
import logging
import pickle
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer 
import time
from scipy import spatial
import math
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV 
import heapq
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
import re


# In[139]:


get_ipython().run_line_magic('writefile', './dupakq.py')


# In[61]:


pd.set_option("display.max_colwidth",None)


# # load data

# In[10]:


df_trampil=pd.read_excel("data/butir_tenaga_trampil.xlsx")#,sep="\t",engine="python") df21=df21[df21.IP.notna()]
df_trampil.head(2)


# In[11]:


df_ahli=pd.read_excel("data/butir_tenaga_ahli.xlsx")#,sep="\t",engine="python") df21=df21[df21.IP.notna()]
df_ahli.head(2)


# In[12]:


df_penunjang=pd.read_excel("data/butir_penunjang.xlsx")#,sep="\t",engine="python") df21=df21[df21.IP.notna()]
df_penunjang.head(4)


# In[14]:


df_penunjang.columns


# In[63]:


df_penunjang[df_penunjang['Unnamed: 5'].notna()].head(2)


# # merge activity details

# In[86]:





# In[135]:


# loop the rows
# combine sub_unsur','kegiatan','Unnamed: 5', 'Unnamed: 6'--> save to new col:activity
# if ak not na then merge previos activies with recent activies   
arr=['nomor_unsur', 'unsur', 'nomor_sub_unsur', 'sub_unsur', 'kegiatan',
       'Unnamed: 5', 'Unnamed: 6']
parent_dict={}
actvities_to_merge=["sub_unsur",'kegiatan','Unnamed: 5', 'Unnamed: 6']
df_penunjang["activities"]=""
previous_activity=""
for idx,row in df_penunjang.iterrows():
    merged=[]
    result=""
    print("ak:{}".format(row.ak))
    try:
        for act in arr:
            if pd.isna(row[act]): #if NaN or ak colum, skip
                if act in parent_dict:
                    row[act]=parent_dict[act]
                continue
            else:
                txt=str(row[act])
                parent_dict[act]=txt
#                 words=re.findall("\w{4}",txt)
#                 if words:
#                     merged.append(txt)
#         result=". ".join(merged)
        if not pd.isna(row["ak"]):
            # take only words with length more than 3 letters
            result=". ".join([v for v in parent_dict.values() if len(v)>3]) 
            for k,v in parent_dict.items():
                row[k]=v
            row.activities=result
            parent_dict.clear()
    except Exception as x:
        print("error at idx:{}-->{}".format(idx,str(x)))


# In[118]:


# df_penunjang["activities"]=df_penunjang[['sub_unsur','kegiatan','Unnamed: 5', 'Unnamed: 6','ak']]\
# .apply(merge_activities,axis=1)


# In[131]:



# df_penunjang[~df_penunjang.ak.isna()][['nomor_unsur', 'unsur', \
#                                        "nomor_sub_unsur",'sub_unsur','kegiatan','Unnamed: 5',\
#                                        'Unnamed: 6å',"activities","ak"]]


# In[133]:



df_penunjang[~df_penunjang.ak.isna()][["activities","ak"]]


# In[134]:


df_penunjang.to_csv("data/butir_penunjang.csv",sep=";")

