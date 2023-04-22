#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn as sk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#Mbledhja e te dhenave
data = open('D:\Desktop\MASTER2\Machine Learning\weatherHistory.csv')


# In[3]:


traindata = pd.read_csv(data)


# In[4]:


traindata


# In[5]:


traindata.shape


# In[6]:


traindata.info()


# In[7]:


traindata.columns


# In[8]:


#Definimi i tipeve te te dhenave
traindata.dtypes


# In[9]:


#Reduktimi i dimensionit - trajtimi i vlerave Null
traindata.describe()


# In[10]:


traindata.describe(include = "all")


# In[11]:


traindata.isnull().sum().sort_values(ascending = False)


# In[12]:


traindata.shape


# In[13]:


traindata=traindata.dropna()


# In[14]:


traindata.shape


# In[15]:


traindata


# In[1]:


#Naive bayes - Gaussian
#Naive bayes - Bernoulli


# In[ ]:




