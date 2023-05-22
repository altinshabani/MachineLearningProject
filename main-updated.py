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
data = open('D:\Desktop\MASTER2\Machine Learning\weatherHistory.csv')


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


# In[25]:


traindata['Temperature(C)'].value_counts()


# In[26]:


traindata['ApparentTemperature(C)'].value_counts()


# In[27]:


# Detektimi dhe perjashtimi i Outliers


# In[28]:


import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize=(25,10))
plt.subplot(1,16,1)
sns.distplot(traindata['Temperature(C)'])
plt.subplot(1,16,2)
sns.distplot(traindata['ApparentTemperature(C)'])
plt.subplot(1,16,3)
sns.distplot(traindata['Humidity'])
plt.subplot(1,16,4)
sns.distplot(traindata['WindSpeed(km/h)'])
plt.subplot(1,16,5)
sns.distplot(traindata['WindBearing(degrees)'])
plt.subplot(1,16,6)
sns.distplot(traindata['Visibility(km)'])
plt.subplot(1,16,7)
sns.distplot(traindata['Pressure(millibars)'])

plt.show()


# In[29]:


# 1. Atributi - Temperature(C)


# In[30]:


print("Highest allowed",traindata['Temperature(C)'].mean() + 3*traindata['Temperature(C)'].std())
print("Lowest allowed",traindata['Temperature(C)'].mean() - 3*traindata['Temperature(C)'].std())


# In[31]:


traindata[(traindata['Temperature(C)'] > 40.00) | (traindata['Temperature(C)'] < -16.77)]


# In[32]:


traindata = traindata[(traindata['Temperature(C)'] < 40.00) & (traindata['Temperature(C)'] > -16.77)]
traindata.shape


# In[33]:


# 2. Atributi - ApparentTemperature(C)


# In[34]:


print("Highest allowed",traindata['ApparentTemperature(C)'].mean() + 3*traindata['ApparentTemperature(C)'].std())
print("Lowest allowed",traindata['ApparentTemperature(C)'].mean() - 3*traindata['ApparentTemperature(C)'].std())


# In[35]:


traindata[(traindata['ApparentTemperature(C)'] > 42.97) | (traindata['ApparentTemperature(C)'] < -21.22)]


# In[36]:


traindata = traindata[(traindata['ApparentTemperature(C)'] < 42.97) & (traindata['ApparentTemperature(C)'] > -21.22)]
traindata.shape


# In[37]:


# 3. Atributi - Humidity


# In[38]:


print("Highest allowed",traindata['Humidity'].mean() + 3*traindata['Humidity'].std())
print("Lowest allowed",traindata['Humidity'].mean() - 3*traindata['Humidity'].std())


# In[39]:


traindata[(traindata['Humidity'] > 1.32) | (traindata['Humidity'] < 0.14)]


# In[40]:


traindata = traindata[(traindata['Humidity'] < 1.32) & (traindata['Humidity'] > 0.14)]
traindata.shape


# In[41]:


# 4. Atributi - WindSpeed(km/h)


# In[42]:


print("Highest allowed",traindata['WindSpeed(km/h)'].mean() + 3*traindata['WindSpeed(km/h)'].std())
print("Lowest allowed",traindata['WindSpeed(km/h)'].mean() - 3*traindata['WindSpeed(km/h)'].std())


# In[43]:


traindata[(traindata['WindSpeed(km/h)'] > 31.56) | (traindata['WindSpeed(km/h)'] < -9.95)]


# In[44]:


traindata = traindata[(traindata['WindSpeed(km/h)'] < 31.56) & (traindata['WindSpeed(km/h)'] > -9.95)]
traindata.shape


# In[45]:


# 5. Atributi - WindBearing(degrees)


# In[46]:


print("Highest allowed",traindata['WindBearing(degrees)'].mean() + 3*traindata['WindBearing(degrees)'].std())
print("Lowest allowed",traindata['WindBearing(degrees)'].mean() - 3*traindata['WindBearing(degrees)'].std())


# In[47]:


traindata[(traindata['WindBearing(degrees)'] > 508.88) | (traindata['WindBearing(degrees)'] < -135.07)]


# In[48]:


traindata = traindata[(traindata['WindBearing(degrees)'] < 508.88) & (traindata['WindBearing(degrees)'] > -135.07)]
traindata.shape


# In[49]:


# 6. Atributi - Visibility(km)


# In[50]:


print("Highest allowed",traindata['Visibility(km)'].mean() + 3*traindata['Visibility(km)'].std())
print("Lowest allowed",traindata['Visibility(km)'].mean() - 3*traindata['Visibility(km)'].std())


# In[51]:


traindata[(traindata['Visibility(km)'] > 22.91) | (traindata['Visibility(km)'] < -2.19)]


# In[52]:


traindata = traindata[(traindata['Visibility(km)'] < 22.91) & (traindata['Visibility(km)'] > -2.19)]
traindata.shape


# In[53]:


# 7. Atributi - Pressure(millibars)


# In[54]:


print("Highest allowed",traindata['Pressure(millibars)'].mean() + 3*traindata['Pressure(millibars)'].std())
print("Lowest allowed",traindata['Pressure(millibars)'].mean() - 3*traindata['Pressure(millibars)'].std())


# In[55]:


traindata[(traindata['Pressure(millibars)'] > 1352.11) | (traindata['Pressure(millibars)'] < 654.81)]


# In[56]:


traindata = traindata[(traindata['Pressure(millibars)'] < 1352.11) & (traindata['Pressure(millibars)'] > 654.81)]
traindata.shape


# In[57]:


# Top 10 vlerat me te medha te temperatures maksimale 
vlera = traindata.sort_values(['Temperature(C)'], ascending=False)
vlera.head(10)


# In[58]:


# Numri i diteve me me shume se 10 ore diellore ne dite
numri= traindata.query('(Humidity > 0.9)')
numri.head()


# In[59]:


# Top 10 vlerat me te uleta te temperatures minimale 
vlera = traindata.sort_values(['Temperature(C)'], ascending=True)
vlera.head(10)


# In[60]:


sns.catplot(x = 'DailySummary', kind = 'count', data = traindata)


# In[61]:


sns.catplot(x = 'PrecipType', kind = 'count', data = traindata)


# In[62]:


traindata['PrecipType'].value_counts()


# In[63]:


plt.figure(figsize=(17,15))
correlacao = sns.heatmap(traindata.corr(), square=True, annot=True)
correlacao.set_xticklabels(correlacao.get_xticklabels(), rotation=90)          
plt.show()


# In[64]:


# Checking the correlation of our MaxTemp and MinTemp variables, we can see that when we have high temperatures, it's
# more common that it doesn't rain in the nextday.

sns.relplot(x='Humidity', y = 'Temperature(C)', hue = "PrecipType", data = traindata)


# In[65]:


# Checking the correlation of our MaxTemp and Sunshine variables, we can see that when we have high temperatures in 
# Sunshine, it is more common that it doesn't rain in the nextday

sns.relplot(x='WindSpeed(km/h)', y = 'Temperature(C)', hue = "PrecipType", data = traindata)


# In[66]:


# Ndertimi i modelit


# In[67]:


# largojme kolonen date pasi qe nuk ndikon asgje ne parashikim
traindata = traindata.drop('FormattedDate', 1)


# In[68]:


traindata


# In[69]:


traindata.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9]]


# In[344]:


X = traindata.iloc[:, [0, 2, 5]]


# In[ ]:





# In[ ]:





# In[345]:


y = traindata.iloc[:, 1].values


# In[346]:


y


# In[347]:


# There are numeric and categorical columns in the data


# In[348]:


numeric_col=traindata.select_dtypes(include="float64").columns
numeric_col


# In[349]:


cat_col=traindata.select_dtypes(include="object").columns
cat_col


# In[350]:


# Using LabelEncoder to transform categorical variables into continuous variables.


# In[351]:


from sklearn.preprocessing import LabelEncoder


# In[352]:


label_encoder_Summary = LabelEncoder()
label_encoder_DailySummary = LabelEncoder()


# In[353]:


X.iloc[:,0] = label_encoder_Summary.fit_transform(X.iloc[:,0])


# In[354]:


X.iloc[:,8]


# In[355]:


X.iloc[:,0] = label_encoder_Summary.fit_transform(X.iloc[:,0])
X.iloc[:,8] = label_encoder_DailySummary.fit_transform(X.iloc[:,8])


# In[356]:


# We will run the models before scaling the data, after we will back here to run the StandardScaler 
# and MinMax Scaler and verify if we have best results using the scaling.


# In[357]:


# StandardScaler


# In[358]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_standard = scaler.fit_transform(X)


# In[359]:


X_standard[0]


# In[360]:


# MinMax Scaler


# In[361]:


from sklearn.preprocessing import MinMaxScaler


# In[362]:


obj_norm = MinMaxScaler().fit(X)


# In[363]:


X


# In[364]:


X_normalization = obj_norm.transform(X)


# In[365]:


X_normalization[0]


# In[366]:


# Transforming Data into Train and Test, here we will use 30% of our data to test the machine learning models.


# In[367]:


from sklearn.model_selection import train_test_split


# In[368]:


x_train, x_test, y_train, y_test = train_test_split(X_normalization, y, test_size = 0.3, random_state = 0)


# In[369]:


x_train.shape, y_train.shape


# In[370]:


x_test.shape, y_test.shape


# In[371]:


# Naive Bayes


# In[372]:


# Here we will use the Naive Bayes Model, we will test Gaussian and Bernoulli models, 
# using our Normal Data, StandardScaler Data and MinMax Data.


# In[373]:


# Running Gaussian Model


# In[374]:


from sklearn.naive_bayes import GaussianNB


# In[375]:


naive_bayes = GaussianNB()
naive_bayes.fit(x_train, y_train)


# In[376]:


predictions = naive_bayes.predict(x_test)


# In[377]:


predictions


# In[378]:


confusion = confusion_matrix(y_test, predictions)


# In[379]:


accuracy_score(y_test, predictions)


# In[380]:


from sklearn.utils.metaestimators import available_if


# In[381]:


from yellowbrick.classifier import ConfusionMatrix


# In[382]:


cm = ConfusionMatrix(naive_bayes)
cm.fit(x_train, y_train)
cm.score(x_test, y_test)


# In[383]:


print(classification_report(y_test, predictions))


# In[384]:


# Running Bernoulli  Model


# In[385]:


from sklearn.naive_bayes import BernoulliNB


# In[386]:


naive_bayes = BernoulliNB()
naive_bayes.fit(x_train, y_train)


# In[387]:


predictions = naive_bayes.predict(x_test)


# In[388]:


predictions


# In[389]:


accuracy_score(y_test, predictions)


# In[390]:


cm = ConfusionMatrix(naive_bayes)
cm.fit(x_train, y_train)
cm.score(x_test, y_test)


# In[391]:


print(classification_report(y_test, predictions))


# In[392]:


# Decision Tree


# In[393]:


# Entropy Calculation


# In[394]:


arvore_entropy = DecisionTreeClassifier(criterion = 'entropy')


# In[395]:


arvore_entropy.fit(x_train, y_train)


# In[396]:


predictions = arvore_entropy.predict(x_test)


# In[397]:


predictions


# In[398]:


accuracy_score(y_test, predictions)


# In[399]:


cm = ConfusionMatrix(arvore_entropy)
cm.fit(x_train, y_train)
cm.score(x_test, y_test)


# In[400]:


# Gini Calculation


# In[401]:


arvore_gini = DecisionTreeClassifier()


# In[402]:


arvore_gini.fit(x_train, y_train)


# In[403]:


predictions = arvore_gini.predict(x_test)


# In[404]:


predictions


# In[405]:


accuracy_score(y_test, predictions)


# In[406]:


cm = ConfusionMatrix(arvore_gini)
cm.fit(x_train, y_train)
cm.score(x_test, y_test)


# In[407]:


# Random Forest


# In[408]:


# Entropy calculation


# In[409]:


from sklearn.ensemble import RandomForestClassifier


# In[410]:


random_forest = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
random_forest.fit(x_train, y_train)


# In[411]:


predictions = random_forest.predict(x_test)


# In[412]:


predictions


# In[413]:


accuracy_score(y_test, predictions)


# In[414]:


cm = ConfusionMatrix(random_forest)
cm.fit(x_train, y_train)
cm.score(x_test, y_test)


# In[415]:


# Gini calculation


# In[416]:


random_forest = RandomForestClassifier(n_estimators = 100, random_state = 0)
random_forest.fit(x_train, y_train)


# In[417]:


predictions = random_forest.predict(x_test)


# In[418]:


predictions


# In[419]:


saktesia = accuracy_score(y_test, predictions)
saktesia = round(saktesia*100, 2)
print(f'Saktesia: {saktesia} %')


# In[420]:


from sklearn.neighbors import KNeighborsClassifier


# In[421]:


knn = KNeighborsClassifier(n_neighbors = 27, metric = 'minkowski', p = 2)


# In[422]:


knn.fit(x_train, y_train)


# In[423]:


predictions = knn.predict(x_test)


# In[424]:


predictions


# In[425]:


accuracy_score(y_test, predictions)


# In[426]:


cm = ConfusionMatrix(knn)
cm.fit(x_train, y_train)
cm.score(x_test, y_test)


# In[427]:


from sklearn.linear_model import LogisticRegression


# In[428]:


logistic = LogisticRegression(random_state = 1, solver='lbfgs', max_iter=1000)


# In[338]:


logistic.fit(x_train, y_train)


# In[339]:


predictions = logistic.predict(x_test)
predictions


# In[340]:


accuracy_score(y_test, predictions)


# In[341]:


cm = ConfusionMatrix(logistic)
cm.fit(x_train, y_train)
cm.score(x_test, y_test)


# In[342]:


# Conclusion


# In[343]:


# Looking at our Data Analysis, we can see that the temperature of the previous day
# is a important variable to get our variable target.
# The best model we have when we analyze the precision is the: ??? model 
# using the ??? calculation, in which we get precision ??? %


# In[ ]:




