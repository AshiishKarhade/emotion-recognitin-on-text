#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('text_emotion.csv')


# In[3]:


# In[4]:


data = data.drop('author', axis=1)

# In[6]:


print(data.shape)


# In[7]:


print(data.describe())


# In[8]:

# In[9]:


data['content'] = data['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))

# In[11]:


data['content'] = data['content'].str.replace('[^\w\s]',' ')


# In[12]:


from nltk.corpus import stopwords
stop = stopwords.words('english')
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[14]:


from textblob import Word
data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# In[15]:

#OPTIONAL

#Correcting Letter Repetitions
import re
def de_repeat(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)
#%%
data['content'] = data['content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))


# In[16]:

# In[17]:

#REmoves least 10000 occurences of words
freq = pd.Series(' '.join(data['content']).split()).value_counts()[-10000:]


# In[18]:


print(type(freq))


# In[19]:


freq = list(freq.index)
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))


# In[20]:


#Encoding output labels 'sadness' as '1' & 'happiness' as '0'
from sklearn import preprocessing
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.sentiment.values)
# Splitting into training and testing data in 90:10 ratio
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data.content.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)


# In[ ]:





# In[21]:


#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.fit_transform(X_val)


# In[ ]:





# In[22]:


# Extracting Count Vectors Parameters
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(analyzer='word')
count_vect.fit(data['content'])
X_train_count =  count_vect.transform(X_train)
X_val_count =  count_vect.transform(X_val)


# In[ ]:





# In[23]:


from sklearn.metrics import accuracy_score


# In[24]:


#MODEL 1: Multinomial NB

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(X_train_tfidf, y_train)

y_nb_pred = nb.predict(X_val_tfidf)
print('naive bayes tfidf accuracy %s' % accuracy_score(y_nb_pred, y_val))


# In[25]:


#MODEL 2: Linear SVM
from sklearn.linear_model import SGDClassifier
svc = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)

svc.fit(X_train_tfidf, y_train)

y_svc_pred = svc.predict(X_val_tfidf)
print('SVC tfidf accuracy %s' % accuracy_score(y_svc_pred, y_val))


# In[26]:


# Model 3: logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1)
logreg.fit(X_train_tfidf, y_train)
y_log_pred = logreg.predict(X_val_tfidf)
print('log reg tfidf accuracy %s' % accuracy_score(y_log_pred, y_val))


# In[27]:


# Model 4: Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train_tfidf, y_train)
y_rf_pred = rf.predict(X_val_tfidf)
print('random forest tfidf accuracy %s' % accuracy_score(y_rf_pred, y_val))


# In[ ]:


# Model 1: Multinomial Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_count, y_train)
y_pred = nb.predict(X_val_count)
print('naive bayes count vectors accuracy %s' % accuracy_score(y_pred, y_val))

print("----------------------------------------------")
# Model 2: Linear SVM
from sklearn.linear_model import SGDClassifier
lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
lsvm.fit(X_train_count, y_train)
y_pred = lsvm.predict(X_val_count)
print('lsvm using count vectors accuracy %s' % accuracy_score(y_pred, y_val))

print("----------------------------------------------")
# Model 3: Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1)
logreg.fit(X_train_count, y_train)
y_pred = logreg.predict(X_val_count)
print('log reg count vectors accuracy %s' % accuracy_score(y_pred, y_val))

print("----------------------------------------------")
# Model 4: Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train_count, y_train)
y_pred = rf.predict(X_val_count)
print('random forest with count vectors accuracy %s' % accuracy_score(y_pred, y_val))


# In[ ]:
