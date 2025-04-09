#!/usr/bin/env python
# coding: utf-8

# # Import the code

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


#  # Load the Dataset

# In[2]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# #  Visualize the Data

# In[3]:


plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()


# # Normalize the Data

# In[9]:


x_train = x_train / 255.0
x_test = x_test / 255.0


# # One-Hot Encode the Labels

# In[5]:


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# # Build the Model

# In[7]:


model = Sequential([
    Flatten(input_shape=(28, 28)),       # Convert 28x28 into 784
    Dense(128, activation='relu'),       # Hidden layer
    Dense(10, activation='softmax')      # Output layer (10 classes)
])


# # Compile the Model

# In[10]:


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# # Evaluate the Model

# In[11]:


test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)


# # Predict

# In[17]:


predictions = model.predict(x_test)

# Show prediction for first test image
plt.imshow(x_test[2], cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[2])}")
plt.show()


# In[ ]:




