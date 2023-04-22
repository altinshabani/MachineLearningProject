#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yellowbrick


# In[2]:


import pandas as pd
import sklearn as sk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pylab
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.classifier import ConfusionMatrix


# In[3]:


#Mbledhja e te dhenave
data = open('C:\\Users\\Image.DESKTOP-P8CTOKN\\Desktop\\MachineLearning\\weatherHistory.csv')


# In[4]:


traindata = pd.read_csv(data)


# In[5]:


traindata


# In[6]:


traindata.shape


# In[7]:


traindata.head()


# In[8]:


traindata.tail()


# In[9]:


traindata.head(10)


# In[10]:


traindata.tail(10)


# In[11]:


traindata.info()


# In[12]:


traindata.Summary.unique()


# In[13]:


traindata.columns


# In[14]:


#Definimi i tipeve te te dhenave
traindata.dtypes


# In[15]:


#Reduktimi i dimensionit - trajtimi i vlerave Null
traindata.describe()


# In[16]:


traindata.describe(include = "all")


# In[17]:


traindata.corr()


# In[18]:


traindata.isnull().sum().sort_values(ascending = False)


# In[19]:


traindata.shape


# In[20]:


traindata=traindata.dropna()
traindata


# In[21]:


traindata.shape


# In[22]:


traindata.duplicated()


# In[23]:


traindata = traindata.drop_duplicates() 
traindata


# In[24]:


traindata.shape


# In[26]:


traindata['Temperature (C)'].value_counts()


# In[27]:


traindata['Apparent Temperature (C)'].value_counts()


# In[ ]:


# Detektimi dhe perjashtimi i Outliers


# In[30]:


import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(25,10))
plt.subplot(1,16,1)
sns.distplot(traindata['Temperature (C)'])
plt.subplot(1,16,2)
sns.distplot(traindata['Apparent Temperature (C)'])
plt.subplot(1,16,3)
sns.distplot(traindata['Humidity'])
plt.subplot(1,16,4)
sns.distplot(traindata['Wind Speed (km/h)'])
plt.subplot(1,16,5)
sns.distplot(traindata['Wind Bearing (degrees)'])
plt.subplot(1,16,6)
sns.distplot(traindata['Visibility (km)'])
plt.subplot(1,16,7)
sns.distplot(traindata['Pressure (millibars)'])

plt.show()


# In[31]:


# 1. Atributi - Temperature(C)


# In[35]:


print("Highest allowed",traindata['Temperature (C)'].mean() + 3*traindata['Temperature (C)'].std())
print("Lowest allowed",traindata['Temperature (C)'].mean() - 3*traindata['Temperature (C)'].std())


# In[36]:


traindata[(traindata['Temperature (C)'] > 40.00) | (traindata['Temperature (C)'] < -16.77)]


# In[37]:


traindata = traindata[(traindata['Temperature (C)'] < 40.00) & (traindata['Temperature (C)'] > -16.77)]
traindata.shape


# In[38]:


# 2. Atributi - ApparentTemperature(C)


# In[39]:


print("Highest allowed",traindata['Apparent Temperature (C)'].mean() + 3*traindata['Apparent Temperature (C)'].std())
print("Lowest allowed",traindata['Apparent Temperature (C)'].mean() - 3*traindata['Apparent Temperature (C)'].std())


# In[40]:


traindata[(traindata['Apparent Temperature (C)'] > 42.97) | (traindata['Apparent Temperature (C)'] < -21.22)]


# In[41]:


traindata = traindata[(traindata['Apparent Temperature (C)'] < 42.97) & (traindata['Apparent Temperature (C)'] > -21.22)]
traindata.shape


# In[42]:


# 3. Atributi - Humidity


# In[43]:


print("Highest allowed",traindata['Humidity'].mean() + 3*traindata['Humidity'].std())
print("Lowest allowed",traindata['Humidity'].mean() - 3*traindata['Humidity'].std())


# In[44]:


traindata[(traindata['Humidity'] > 1.32) | (traindata['Humidity'] < 0.14)]


# In[45]:


traindata = traindata[(traindata['Humidity'] < 1.32) & (traindata['Humidity'] > 0.14)]
traindata.shape


# In[46]:


# 4. Atributi - WindSpeed(km/h)


# In[47]:


print("Highest allowed",traindata['Wind Speed (km/h)'].mean() + 3*traindata['Wind Speed (km/h)'].std())
print("Lowest allowed",traindata['Wind Speed (km/h)'].mean() - 3*traindata['Wind Speed (km/h)'].std())


# In[48]:


traindata[(traindata['Wind Speed (km/h)'] > 31.56) | (traindata['Wind Speed (km/h)'] < -9.95)]


# In[49]:


traindata = traindata[(traindata['Wind Speed (km/h)'] < 31.56) & (traindata['Wind Speed (km/h)'] > -9.95)]
traindata.shape


# In[50]:


# 5. Atributi - WindBearing(degrees)


# In[51]:


print("Highest allowed",traindata['Wind Bearing (degrees)'].mean() + 3*traindata['Wind Bearing (degrees)'].std())
print("Lowest allowed",traindata['Wind Bearing (degrees)'].mean() - 3*traindata['Wind Bearing (degrees)'].std())


# In[52]:


traindata[(traindata['Wind Bearing (degrees)'] > 508.88) | (traindata['Wind Bearing (degrees)'] < -135.07)]


# In[53]:


traindata = traindata[(traindata['Wind Bearing (degrees)'] < 508.88) & (traindata['Wind Bearing (degrees)'] > -135.07)]
traindata.shape


# In[54]:


# 6. Atributi - Visibility(km)


# In[55]:


print("Highest allowed",traindata['Visibility (km)'].mean() + 3*traindata['Visibility (km)'].std())
print("Lowest allowed",traindata['Visibility (km)'].mean() - 3*traindata['Visibility (km)'].std())


# In[56]:


traindata[(traindata['Visibility (km)'] > 22.91) | (traindata['Visibility (km)'] < -2.19)]


# In[57]:


traindata = traindata[(traindata['Visibility (km)'] < 22.91) & (traindata['Visibility (km)'] > -2.19)]
traindata.shape


# In[58]:


# 7. Atributi - Pressure(millibars)


# In[59]:


print("Highest allowed",traindata['Pressure (millibars)'].mean() + 3*traindata['Pressure (millibars)'].std())
print("Lowest allowed",traindata['Pressure (millibars)'].mean() - 3*traindata['Pressure (millibars)'].std())


# In[60]:


traindata[(traindata['Pressure (millibars)'] > 1352.11) | (traindata['Pressure (millibars)'] < 654.81)]


# In[61]:


traindata = traindata[(traindata['Pressure (millibars)'] < 1352.11) & (traindata['Pressure (millibars)'] > 654.81)]
traindata.shape


# In[62]:


# Top 10 vlerat me te medha te temperatures maksimale 
vlera = traindata.sort_values(['Temperature (C)'], ascending=False)
vlera.head(10)


# In[63]:


# Numri i diteve me me shume se 10 ore diellore ne dite
numri= traindata.query('(Humidity > 0.9)')
numri.head()


# In[64]:


# Top 10 vlerat me te uleta te temperatures minimale 
vlera = traindata.sort_values(['Temperature (C)'], ascending=True)
vlera.head(10)


# In[66]:


sns.catplot(x = 'Daily Summary', kind = 'count', data = traindata)


# In[67]:


sns.catplot(x = 'Precip Type', kind = 'count', data = traindata)


# In[68]:


traindata['Precip Type'].value_counts()


# In[69]:


plt.figure(figsize=(17,15))
correlacao = sns.heatmap(traindata.corr(), square=True, annot=True)
correlacao.set_xticklabels(correlacao.get_xticklabels(), rotation=90)          
plt.show()


# In[70]:


# Checking the correlation of our MaxTemp and MinTemp variables, we can see that when we have high temperatures, it's
# more common that it doesn't rain in the nextday.

sns.relplot(x='Humidity', y = 'Temperature (C)', hue = "Precip Type", data = traindata)


# In[ ]:




