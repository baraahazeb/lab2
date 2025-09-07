#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


df=pd.read_csv(r"C:\Users\IT\Desktop\,,,\braaa\baraa.csv")


# In[5]:


df.head()


# In[7]:


df['statement']=df['status'].apply(lambda x : 1 if x=='statement' else 0)
df


# In[8]:


dataset =df.drop('status',axis='columns')
dataset


# In[9]:


df.groupby('status').describe()


# In[10]:


from sklearn.model_selection import train_test_split
train,test =train_test_split(dataset)
train.shape


# In[11]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


vectorizer = CountVectorizer()
train_vector =vectorizer.fit_transform(train.Message.values)
train_vector.toarray()


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(train_vector,train.spam)


# In[ ]:


test_vector =vectorizer.fit_transform(test.Message.values)
test_vector.toarray()

