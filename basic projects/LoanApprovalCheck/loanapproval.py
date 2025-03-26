#!/usr/bin/env python
# coding: utf-8

# In[2]:


#download scikit learn 
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


# In[3]:


from sklearn.ensemble import RandomForestRegressor


# In[4]:


from sklearn.model_selection import train_test_split


# In[9]:


data = pd.read_csv('loanapproval.csv')


# In[10]:


data.columns


# In[11]:


data.head()


# In[12]:


features = [' no_of_dependents',' loan_term',' cibil_score',' residential_assets_value',' commercial_assets_value', ' luxury_assets_value', ' bank_asset_value',]


# In[28]:


data[' loan_status'] = data[' loan_status'].apply(lambda x: 1 if str(x).lower() == 'approved' else 0)

data = data.dropna(axis=0)


# In[29]:


y = data[' loan_status']


# In[30]:


x = data[features]


# In[31]:


#making the first model using decision tree
model1 = DecisionTreeRegressor(random_state=1)


# In[32]:


model2 = RandomForestRegressor(random_state=1)


# In[39]:


trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.2, random_state=42)


# In[40]:


model1.fit(trainx,trainy)


# In[41]:


model2.fit(trainx,trainy)


# In[42]:


model1.predict(testx)


# In[43]:


model2.predict(testx)


# In[44]:


from sklearn.metrics import mean_absolute_error


# In[45]:


y_pred = model1.predict(testx)


# In[46]:


mae = mean_absolute_error(testy, y_pred)
print("Mean Absolute Error:", mae)


# In[47]:


y_pred


# In[ ]:




