#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np 
import pandas as pd 
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=15)
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)
import warnings
warnings.simplefilter(action='ignore')


# In[26]:


mydata = pd.read_csv("D:\\445\\khidmahactuall.csv")
# preview train data
mydata.head()


# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
columns = ["Age","Smoke","Dia_family_mem","Height","Weight","Pulse","Bp","Date of detection","FBS","ABF","T.Chol","TG","SGPT","Creatinine"] 
X = mydata[columns]
y = mydata['Diagnosis']
mymodel = LogisticRegression()
rfe = RFE(mymodel, 5)
rfe = rfe.fit(X, y)
print('Selected features are as follows: %s' % list(X.columns[rfe.support_]))


# In[28]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
X = mydata[columns]
y = mydata['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
y_pred_proba = logreg.predict_proba(X_test)[:, 1]
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy = %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss = %2.3f" % log_loss(y_test, y_pred_proba))


# In[29]:


logreg = LogisticRegression()
scores_accuracy = cross_val_score(logreg, X, y, cv=11, scoring='accuracy')
scores_log_loss = cross_val_score(logreg, X, y, cv=11, scoring='neg_log_loss')
scores_auc = cross_val_score(logreg, X, y, cv=11, scoring='roc_auc')
print('K-fold cross-validation results are as follows:')
print(logreg.__class__.__name__+" average accuracy = %2.3f" % scores_accuracy.mean())
print(logreg.__class__.__name__+" average log_loss = %2.3f" % -scores_log_loss.mean())
print(logreg.__class__.__name__+" average auc = %2.3f" % scores_auc.mean())


# In[30]:


from sklearn.model_selection import cross_validate
scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc'}
Crossvalidate = LogisticRegression()
res = cross_validate(Crossvalidate, X, y, cv=11, scoring=list(scoring.values()), 
                         return_train_score=False)
print('K-fold cross-validation results after using cross_validate() function:')
for score in range(len(scoring)):
    print(Crossvalidate.__class__.__name__+" average %s: %.3f (+/-%.4f)" % (list(scoring.keys())[score], -res['test_%s' % list(scoring.values())[score]].mean()
                               if list(scoring.values())[score]=='neg_log_loss' 
                               else res['test_%s' % list(scoring.values())[score]].mean(), 
                               res['test_%s' % list(scoring.values())[score]].std()))


# In[ ]:





# In[ ]:




