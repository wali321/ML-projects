#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[4]:


data = pd.read_csv('melb_data.csv')


# In[5]:


data = data.dropna(axis=0)


# In[7]:


y = data.Price


# In[8]:


features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']


# In[9]:


X = data[features]


# In[11]:


train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)


# In[12]:


from sklearn.ensemble import RandomForestRegressor


# In[13]:


from sklearn.metrics import mean_absolute_error


# In[14]:


forest_model = RandomForestRegressor(random_state=1)


# In[15]:


forest_model.fit(train_X, train_y)


# In[16]:


melb_preds = forest_model.predict(val_X)


# In[17]:


print(mean_absolute_error(val_y, melb_preds))


# In[ ]:




