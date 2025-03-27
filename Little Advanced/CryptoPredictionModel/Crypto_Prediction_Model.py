#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[56]:


# we will be using gradient booster in this project
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[57]:


#loading bitcoin dataset
btc_data = pd.read_csv('coin_Bitcoin.csv')


# In[58]:


btc_data = btc_data.dropna()


# In[59]:


btc_data.columns


# In[60]:


#loading etherium dataset
eth_data = pd.read_csv('coin_Ethereum.csv')


# In[61]:


eth_data = eth_data.dropna()


# In[62]:


eth_data.columns


# In[63]:


crypto_data = pd.concat([btc_data, eth_data], ignore_index=True)


# In[64]:


crypto_data.drop(columns=["SNo", "Date"], inplace=True)


# In[65]:


X = crypto_data.drop(columns=["Close"])  # Predicting 'Close' price
y = crypto_data["Close"]


# In[66]:


categorical_cols = ["Name", "Symbol"]
numeric_cols = ["High", "Low", "Open", "Volume", "Marketcap"]


# In[67]:


preprocessor = ColumnTransformer(transformers=[
    ("num_imputer", SimpleImputer(strategy="mean"), numeric_cols),  # Fill missing values
    ("cat_encoder", OneHotEncoder(handle_unknown="ignore"), categorical_cols)  # Convert categories
])


# In[69]:


#Creating a pipeline with SimpleImputer
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),  # Preprocessing (imputation + encoding)
    ("model", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1))  # Model training
])


# In[70]:


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[71]:


# Train the model
pipeline.fit(X_train, y_train)


# In[72]:


score = pipeline.score(X_test, y_test)
print(f"Model Score: {score:.4f}") 


# In[73]:


#done


# In[74]:


#testing the model 
#not necessary


# In[75]:


import pandas as pd

# Define mock crypto data
mock_data = [
    {"SNo": 1, "Name": "Bitcoin", "Symbol": "BTC", "Date": "2023-01-01", "High": 45000, "Low": 44000, "Open": 44500, "Close": 44800, "Volume": 3000000000, "Marketcap": 850000000000},
    {"SNo": 2, "Name": "Bitcoin", "Symbol": "BTC", "Date": "2023-01-02", "High": 45500, "Low": 44200, "Open": 44800, "Close": 45200, "Volume": 3100000000, "Marketcap": 860000000000},
    {"SNo": 3, "Name": "Bitcoin", "Symbol": "BTC", "Date": "2023-01-03", "High": 46000, "Low": 45000, "Open": 45300, "Close": 45900, "Volume": 3200000000, "Marketcap": 870000000000},
    {"SNo": 4, "Name": "Ethereum", "Symbol": "ETH", "Date": "2023-01-01", "High": 3200, "Low": 3100, "Open": 3150, "Close": 3180, "Volume": 1200000000, "Marketcap": 370000000000},
    {"SNo": 5, "Name": "Ethereum", "Symbol": "ETH", "Date": "2023-01-02", "High": 3300, "Low": 3120, "Open": 3180, "Close": 3270, "Volume": 1250000000, "Marketcap": 380000000000},
    {"SNo": 6, "Name": "Ethereum", "Symbol": "ETH", "Date": "2023-01-03", "High": 3400, "Low": 3200, "Open": 3270, "Close": 3350, "Volume": 1300000000, "Marketcap": 390000000000},
    {"SNo": 7, "Name": "Bitcoin", "Symbol": "BTC", "Date": "2023-01-04", "High": 47000, "Low": 46000, "Open": 46200, "Close": 46900, "Volume": 3300000000, "Marketcap": 880000000000},
    {"SNo": 8, "Name": "Ethereum", "Symbol": "ETH", "Date": "2023-01-04", "High": 3450, "Low": 3250, "Open": 3350, "Close": 3400, "Volume": 1350000000, "Marketcap": 395000000000}
]

# Convert list to DataFrame
df = pd.DataFrame(mock_data)

# Display DataFrame
print(df)


# In[77]:


df.drop(columns=["SNo", "Date"], inplace=True)

# Separate features (X) and target (y)
Xtest = df.drop(columns=["Close"])  # Features (remove 'Close' column)
ytest = df["Close"]  # Actual Close prices for comparison


# In[78]:


# Make predictions using trained pipeline
y_pred = pipeline.predict(X_test)

# Show predictions alongside actual values
results = pd.DataFrame({"Actual Close": y_test, "Predicted Close": y_pred})
print(results)


# In[ ]:




