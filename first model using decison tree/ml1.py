#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor


# In[2]:


data = pd.read_csv("melb_data.csv")


# In[3]:


data.columns


# In[4]:


data = data.dropna(axis=0)


# In[5]:


y = data.Price


# In[6]:


melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']


# In[7]:


X = data[melbourne_features]


# In[8]:


X.describe()


# In[9]:


X.head()


# In[10]:


melbourne_model = DecisionTreeRegressor(random_state=1)


# In[11]:


melbourne_model.fit(X, y)


# In[12]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


# In[ ]:




