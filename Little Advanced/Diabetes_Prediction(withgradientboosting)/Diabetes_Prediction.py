#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# In[3]:


#loading data
data = pd.read_csv('diabetes.csv')


# In[4]:


data.columns


# In[5]:


data.head()


# In[6]:


features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']


# In[7]:


X = data[features]


# In[8]:


y = data['Outcome']


# In[9]:


trainx,valx,trainy,valy = train_test_split(X,y,test_size=0.2,random_state=1)


# In[10]:


pipeline=Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
    ('model',XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))])


# In[11]:


pipeline.fit(trainx,trainy)


# In[12]:


pred=pipeline.predict(valx)


# In[13]:


print(mean_absolute_error(valy, pred))


# In[14]:


pipeline.predict(valx)


# In[15]:


import numpy as np


# In[16]:


arr = np.array([pred])


# In[17]:


binary_outcome = (arr > 0.0).astype(int)


# In[18]:


#our model is complete
print(binary_outcome)


# In[19]:


#lets check test it by using a different dataseyt as testing dataset


# In[20]:


testingdata = pd.read_csv('testdiabetes.csv')
testingdata.columns


# In[21]:


features=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age' ]
Testingx = testingdata[features]
Testingy =testingdata['Outcome']


# In[22]:


test = pipeline.predict(Testingx)


# In[23]:


binarized = np.array([test])


# In[24]:


testbinary_outcome = (binarized > 0.0).astype(int)


# In[25]:


print(testbinary_outcome)


# In[26]:


#done

