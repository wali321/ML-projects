#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers


# In[2]:


data=pd.read_csv("cereal.csv")


# In[3]:


data.columns


# In[4]:


data.head()


# In[5]:


data['type'] = data['type'].map({'C': 0, 'H': 1})


# In[6]:


data = pd.get_dummies(data, columns=['mfr'])


# In[7]:


data.columns


# In[8]:


features = ['type', 'protein', 'fat', 'sodium', 'fiber',
       'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight', 'cups',
       'rating', 'mfr_A', 'mfr_G', 'mfr_K', 'mfr_N', 'mfr_P', 'mfr_Q',
       'mfr_R']


# In[9]:


X = data[features]
y = data[ 'calories']


# In[10]:


from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[21]:


model = keras.Sequential([
    keras.Input(shape=(20,)),         # Input layer for 20 features
    layers.Dense(200, activation='relu'),
    layers.Dense(68, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(1)                   # Output layer (e.g., for predicting rating)
])


# In[26]:


model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=60, validation_split=0.2)


# In[28]:


y_pred = model.predict(X_test)


# In[29]:


print(y_pred[:5])  # First 5 predictions


# In[30]:


comparison = pd.DataFrame({
    'Actual': y_test[:5].values.reshape(-1),
    'Predicted': y_pred[:5].reshape(-1)
})
print(comparison)


# In[ ]:




