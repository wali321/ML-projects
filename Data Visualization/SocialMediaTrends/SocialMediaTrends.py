#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


data = pd.read_csv("vsmt.csv")


# In[4]:


data.columns


# In[5]:


data.head()


# ## Platform Usage

# In[8]:


sns.countplot(data=data, x='Platform')


# ## Content Type Performance

# In[9]:


sns.barplot(data=data, x='Content_Type', y='Likes')


# ## Engagement Level

# In[10]:


sns.boxplot(data=data, x='Region', y='Likes', hue='Engagement_Level')


# In[12]:


sns.heatmap(data[['Views', 'Likes', 'Shares', 'Comments']].corr(), annot=True)


# In[ ]:




