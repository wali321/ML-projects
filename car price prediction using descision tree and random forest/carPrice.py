#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[2]:


dataset = pd.read_csv('carprice.csv') 


# In[3]:


dataset = dataset.dropna(axis=0)


# In[4]:


dataset.columns


# In[5]:


features = [ 'Year', 'Engine Size',
       'Mileage', ]


# In[6]:


y = dataset.Price


# In[7]:


X = dataset[features]


# In[8]:


trainx, valx, trainy, valy = train_test_split(X, y, test_size=0.3, random_state=1)


# In[9]:


model1 = DecisionTreeRegressor(random_state=1)


# In[10]:


model1.fit(trainx,trainy)


# In[11]:


model2 = RandomForestRegressor(random_state=1)


# In[12]:


model2.fit(trainx,trainy)


# In[13]:


model1pred=model1.predict(valx)


# In[14]:


model2pred=model2.predict(valx)


# In[15]:


predictions_df = pd.DataFrame({
    "Actual Price": valy.values,  # Use y_val (not trainx.Price)
    "Model1 Predicted": model1pred,
    "Model2 Predicted": model2pred
})


# In[16]:


gpredictions_df.head()


# In[ ]:




