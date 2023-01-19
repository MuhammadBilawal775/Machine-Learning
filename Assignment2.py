#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd  
import seaborn as sns 
from sklearn.datasets import fetch_openml


# In[2]:


from sklearn.datasets import fetch_openml

opt = fetch_openml(name="optdigits", as_frame=True)


# In[17]:


print(opt.keys())


# In[4]:


opt.target


# In[5]:


opt.data


# In[6]:



my_df = pd.DataFrame(opt.data, columns=opt.feature_names)
my_df.head()


# In[28]:


my_df['input61'] = opt.target
my_df.head()


# In[14]:


my_df.isnull().sum()


# In[29]:


my_df = my_df.dropna(1)
print(my_df.dtypes)


# In[18]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(my_df['input61'], bins=10)
plt.show()


# In[30]:


correlation_matrix = my_df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)


# In[33]:


plt.figure(figsize=(20, 5))

features = ['input7','input29']
target = my_df['input61']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = my_df[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('input61')


# In[35]:


print(my_df.isnull().values.any())


# In[38]:


X = pd.DataFrame(np.c_[my_df['input7'], my_df['input29']], columns = ['input7','input29'])
y = my_df['input61']

print(X.dtypes)
print(y.dtypes)


# In[39]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[40]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[41]:


from sklearn.metrics import r2_score
y_train_predict = lin_model.predict(X_train)
testPred = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (mean_squared_error(Y_test, y_test_predict))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[43]:


#plot to see
plt.scatter(X_train['input7'], Y_train,color='g') 
plt.plot(X_test['input7'], y_test_predict,color='k') 



plt.show()


# In[44]:


plt.scatter(X_test['input7'],Y_test)
plt.scatter(X_test['input7'],y_test_predict)
plt.show()


# In[ ]:




