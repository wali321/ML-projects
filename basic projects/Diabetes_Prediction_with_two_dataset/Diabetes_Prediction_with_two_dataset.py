#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[29]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[12]:


#loading data
data = pd.read_csv('diabetes.csv')


# In[13]:


data.columns


# In[14]:


data.head()


# In[15]:


features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']


# In[16]:


X = data[features]


# In[9]:


y = data['Outcome']


# In[18]:


trainx,valx,trainy,valy = train_test_split(X,y,test_size=0.2,random_state=1)


# In[36]:


model = RandomForestRegressor(n_estimators=10,random_state=1)  


# In[37]:


model.fit(trainx,trainy)


# In[38]:


pred=model.predict(valx)


# In[39]:


print(mean_absolute_error(valy, pred))


# In[41]:


model.predict(valx)


# In[42]:


import numpy as np


# In[43]:


arr = np.array([pred])


# In[60]:


binary_outcome = (arr > 0.0).astype(int)


# In[48]:


#our model is complete
print(binary_outcome)


# In[49]:


#lets check test it by using a different dataseyt as testing dataset


# In[52]:


testingdata = pd.read_csv('testdiabetes.csv')
testingdata.columns


# In[54]:


features=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age' ]
Testingx = testingdata[features]
Testingy =testingdata['Outcome']


# In[56]:


test = model.predict(Testingx)


# In[57]:


binarized = np.array([test])


# In[59]:


testbinary_outcome = (binarized > 0.0).astype(int)


# In[61]:


print(testbinary_outcome)


# In[ ]:




