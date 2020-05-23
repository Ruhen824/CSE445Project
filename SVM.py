#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score


# In[2]:


mydata = pd.read_csv("D:\\445\\khidmahactuall.csv")
mydata.describe()


# In[3]:


trainnow = mydata.drop('Diagnosis', axis=1)
labels = mydata['Diagnosis']


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trainnow, labels, test_size=0.2, random_state=50)


# In[5]:


from sklearn import svm
from sklearn import metrics
svm1 = svm.SVC(verbose=True)
svm1.fit(X_train,y_train)


# In[6]:


y_pred = svm1.predict(X_test)
print('Accuracy Score: {}'.format(metrics.accuracy_score(y_test,y_pred)))
print('ROC AUC Score: {}'.format(metrics.roc_auc_score(y_test,y_pred)))


# In[7]:


from sklearn.model_selection import cross_val_score
n_folds = 11
scores = cross_val_score(svm1, X_train, y_train, cv=n_folds, scoring='roc_auc', n_jobs=-1) 


# In[8]:


fold_names = list(range(n_folds))
fold_names.append('Average')

avg_score = scores.mean()
scores = list(scores)
scores.append(avg_score)
cv_score = pd.DataFrame({'Fold Index': fold_names, 'ROC AUC (Norm)':scores, })

cv_score


# In[9]:


from sklearn.model_selection import StratifiedKFold
strat_scores = cross_val_score(svm1,X_train,y_train,cv=StratifiedKFold(11,random_state=1,shuffle=True),scoring='roc_auc',n_jobs=-1)
avg_strat_score = strat_scores.mean()
strat_scores = list(strat_scores)
strat_scores.append(avg_strat_score)

cv_score['ROC AUC (Strat)'] = strat_scores
cv_score


# In[ ]:




