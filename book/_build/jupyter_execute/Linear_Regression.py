#!/usr/bin/env python
# coding: utf-8

# # Linear Regression

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[2]:


# load the data from scikit-learn (a bunch is returned)
boston = load_boston()
# required components
data = boston["data"]
target = boston["target"]
cols = boston["feature_names"]


# In[3]:


def get_var(df, var_name):
    globals()[var_name] = df


# fmt:off
(
pd.DataFrame(data, columns=cols)
 .assign(target=target)
 .dropna(axis=1)
 .rename(str.lower, axis=1)
 .pipe(get_var, "df1")
 )

Y = df1['target']
X = df1.drop(columns='target',axis=1)


# In[4]:



print(X.shape)
print(Y.shape)


# In[5]:


X_train, Y_train,X_test,Y_test = train_test_split(X, Y,test_size=0.3,random_state=42)

