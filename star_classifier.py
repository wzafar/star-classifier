#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import sklearn as sk
import scipy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, normalize 
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score,plot_confusion_matrix,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[4]:


train = pd.read_csv('../syde522Project-main/data/exoTrain.csv')


# In[6]:


train.head()


# In[7]:


test = pd.read_csv('../syde522Project-main/data/exoTest.csv')


# In[8]:


test.head()


# In[10]:


train.shape


# In[11]:


test.shape


# In[12]:


train.LABEL.value_counts() # 1 represents non-exoplanet-stars and 2 represents exoplanet-stars


# In[13]:


test.LABEL.value_counts()


# In[33]:


# visualizing non-exoplanet-stars
fig = plt.figure(figsize=(15,40))
x = np.array(range(3197))
for i in range(20):
    ax = fig.add_subplot(14,4,i+1)
    ax.scatter(x,train[train.LABEL == 1].iloc[i,1:])


# In[36]:


# visualizing exoplanet-stars: trend over time is approximately periodic and outliers are present
fig = plt.figure(figsize=(15,40))
x = np.array(range(3197))
for i in range(20):
    ax = fig.add_subplot(14,4,i+1)
    ax.scatter(x,train[train.LABEL == 2].iloc[i,1:])


# In[58]:


# visualizing flux distribution of non-exoplanet-stars: narrower distribution b/c flux values are more consistent
fig = plt.figure(figsize=(15,40))
for i in range(20):
    ax = fig.add_subplot(14,4,i+1)
    train[train.LABEL ==1].iloc[i,1:].hist(bins=60)


# In[59]:


# visualizing flux distribution of exoplanet-stars: wider distribution b/c flux values vary more 
fig = plt.figure(figsize=(15,40))
for i in range(20):
    ax = fig.add_subplot(14,4,i+1)
    train[train.LABEL ==2].iloc[i,1:].hist(bins=60)


# In[60]:


X_train = train.drop('LABEL', axis=1)
y_train = train['LABEL'].values
X_test = test.drop('LABEL', axis=1)
y_test = test['LABEL'].values


# In[63]:


# baseline model without any data processing
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
preds = knn.predict(X_test)
print(classification_report(y_test,preds))


# In[65]:


# Another baseline model w/o any processing
from sklearn import svm
svm = svm.SVC(kernel = 'linear')
svm.fit(X_train,y_train)
preds = svm.predict(X_test)
print(classification_report(y_test,preds))


# In[66]:


# Another baseline model w/o any processing
from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt.fit(X_train,y_train)
preds = dt.predict(X_test)
print(classification_report(y_test,preds))


# In[67]:


# Last baseline model w/o any processing
from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()
lgr.fit(X_train,y_train)
preds = lgr.predict(X_test)
print(classification_report(y_test,preds))


# In[68]:


# some observations:
# many outliers
# highly imbalanced data
# denoise signal


# In[ ]:


# handling imbalanced data using smote

