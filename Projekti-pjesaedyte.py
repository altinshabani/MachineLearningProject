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


# In[72]:


# Checking the correlation of our MaxTemp and Sunshine variables, we can see that when we have high temperatures in 
# Sunshine, it is more common that it doesn't rain in the nextday

sns.relplot(x='Wind Speed (km/h)', y = 'Temperature (C)', hue = "Precip Type", data = traindata)


# In[73]:


# Ndertimi i modelit


# In[75]:


# largojme kolonen date pasi qe nuk ndikon asgje ne parashikim
traindata = traindata.drop('Formatted Date', 1)


# In[76]:


traindata


# In[77]:


traindata.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9]]


# In[78]:


X = traindata.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9]]


# In[79]:


y = traindata.iloc[:, 1].values


# In[80]:


y


# In[81]:


# There are numeric and categorical columns in the data


# In[82]:


numeric_col=traindata.select_dtypes(include="float64").columns
numeric_col


# In[83]:


# Using LabelEncoder to transform categorical variables into continuous variables.


# In[84]:


from sklearn.preprocessing import LabelEncoder


# In[85]:


label_encoder_Summary = LabelEncoder()
label_encoder_DailySummary = LabelEncoder()


# In[86]:


X.iloc[:,0] = label_encoder_Summary.fit_transform(X.iloc[:,0])


# In[87]:


X.iloc[:,8]


# In[88]:


X.iloc[:,0] = label_encoder_Summary.fit_transform(X.iloc[:,0])
X.iloc[:,8] = label_encoder_DailySummary.fit_transform(X.iloc[:,8])


# In[89]:


# We will run the models before scaling the data, after we will back here to run the StandardScaler 
# and MinMax Scaler and verify if we have best results using the scaling.


# In[90]:


# StandardScaler


# In[91]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)


# In[92]:


X_standard[0]


# In[93]:


# MinMax Scaler


# In[94]:


from sklearn.preprocessing import MinMaxScaler


# In[95]:


obj_norm = MinMaxScaler().fit(X)


# In[96]:


X


# In[97]:


X_normalization = obj_norm.transform(X)


# In[98]:


X_normalization[0]


# In[99]:


# Transforming Data into Train and Test, here we will use 30% of our data to test the machine learning models.


# In[100]:


from sklearn.model_selection import train_test_split


# In[101]:


x_train, x_test, y_train, y_test = train_test_split(X_normalization, y, test_size = 0.3, random_state = 0)


# In[102]:


x_train.shape, y_train.shape


# In[103]:


x_test.shape, y_test.shape


# In[104]:


# Naive Bayes


# In[105]:


# Here we will use the Naive Bayes Model, we will test Gaussian and Bernoulli models, 
# using our Normal Data, StandardScaler Data and MinMax Data.


# In[106]:


# Running Gaussian Model


# In[107]:


from sklearn.naive_bayes import GaussianNB


# In[108]:


naive_bayes = GaussianNB()
naive_bayes.fit(x_train, y_train)


# In[109]:


predictions = naive_bayes.predict(x_test)


# In[110]:


predictions


# In[111]:


confusion = confusion_matrix(y_test, predictions)


# In[112]:


accuracy_score(y_test, predictions)


# In[113]:


from sklearn.utils.metaestimators import available_if


# In[114]:


from yellowbrick.classifier import ConfusionMatrix


# In[115]:


cm = ConfusionMatrix(naive_bayes)
cm.fit(x_train, y_train)
cm.score(x_test, y_test)


# In[116]:


print(classification_report(y_test, predictions))


# In[117]:


# Running Bernoulli  Model


# In[118]:


from sklearn.naive_bayes import BernoulliNB


# In[119]:


naive_bayes = BernoulliNB()
naive_bayes.fit(x_train, y_train)


# In[120]:


predictions = naive_bayes.predict(x_test)


# In[121]:


predictions


# In[122]:


accuracy_score(y_test, predictions)


# In[123]:


cm = ConfusionMatrix(naive_bayes)
cm.fit(x_train, y_train)
cm.score(x_test, y_test)


# In[124]:


print(classification_report(y_test, predictions))


# In[ ]:




