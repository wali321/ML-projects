#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


data = pd.read_csv('melb_data.csv')


# In[18]:


filtered = data.dropna(axis =0) 
filtered.columns


# In[19]:


y = filtered.Price


# In[20]:


filters = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea','YearBuilt', 'Lattitude', 'Longtitude']


# In[24]:


X = filtered[filters]


# In[25]:


from sklearn.tree import DecisionTreeRegressor


# In[27]:


melbourne_model = DecisionTreeRegressor()

melbourne_model.fit(X, y)


# In[28]:


from sklearn.metrics import mean_absolute_error


# In[29]:


predicted_home_prices = melbourne_model.predict(X)


# In[30]:


mean_absolute_error(y, predicted_home_prices)


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[33]:


melbourne_model = DecisionTreeRegressor()


# In[34]:


melbourne_model.fit(train_X, train_y)


# In[35]:


val_predictions = melbourne_model.predict(val_X)


# In[36]:


print(mean_absolute_error(val_y, val_predictions))


# In[ ]:




