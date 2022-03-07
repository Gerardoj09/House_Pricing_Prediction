#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


df= pd.read_csv('df_ETL.csv')


# The first thing is to return the datasets to the way they were before cleaning to be able to work on them separately.

# In[3]:


df.shape


# In[4]:


df.head()


# 
# Now we will split our  dataset again to have separeted the training set with the testing set

# In[5]:


df.columns


# In[6]:


df = df[['OverallQual', 'YearBuilt', 'YearRemodAdd', 'GrLivArea','GarageCars', 'GarageArea','SalePrice']]


# In[7]:


y = df.SalePrice
X = df.drop(columns = 'SalePrice')


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, shuffle  = True)


# In[9]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[10]:


pickle.dump(model, open('model.pkl', 'wb'))


# In[11]:


y_prediction = model.predict(X_test)
y_prediction


# In[12]:


mean_squared_error(y_test, y_prediction)


# In[13]:


r2_score(y_test, y_prediction)

