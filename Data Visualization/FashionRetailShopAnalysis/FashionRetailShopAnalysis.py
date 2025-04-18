#!/usr/bin/env python
# coding: utf-8

# In[11]:


import seaborn as sns
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("frs.csv")


# In[23]:


# Round the ratings to the nearest integer
data['Rounded Rating'] = data['Review Rating'].round()

# Now plot
sns.countplot(x='Rounded Rating', data=data)
plt.title("Customer Review Ratings")
plt.tight_layout()
plt.show()


# In[24]:


sns.countplot(x='Payment Method', data=data)
plt.title("Preferred Payment Methods")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[25]:


sns.histplot(data['Purchase Amount (USD)'], bins=20, kde=True)
plt.title("Distribution of Purchase Amounts")
plt.tight_layout()
plt.show()


# In[27]:


avg_purchase = data.groupby('Item Purchased')['Purchase Amount (USD)'].mean().sort_values()

avg_purchase.plot(kind='barh', figsize=(10, 10))
plt.title("Average Purchase Amount by Item")
plt.xlabel("USD")
plt.tight_layout()
plt.show()


# In[33]:


plt.figure(figsize=(10, 15))  # Width x Height (increase height for more y-labels)

sns.countplot(
    y='Item Purchased',
    data=data,
    order=data['Item Purchased'].value_counts().index,

)

plt.title("Most Frequently Purchased Items", fontsize=16)
plt.xlabel("Count")
plt.ylabel("Item Purchased")
plt.tight_layout()
plt.show()


# In[ ]:




