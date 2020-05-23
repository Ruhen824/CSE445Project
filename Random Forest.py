#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from urllib.request import urlopen 
plt.style.use('ggplot')
pd.set_option('display.max_columns', 500)


# In[2]:


mydata = pd.read_csv('D:\\445\\khidmahactuall.csv')
names = ["Age","Smoke","Dia_family_mem","Height","Weight","Pulse","Bp","Date of detection","FBS","ABF","T.Chol","TG","SGPT","Creatinine"] 


# In[3]:


space = mydata.iloc[:, mydata.columns != 'Diagnosis']
predict_class = mydata.iloc[:, mydata.columns == 'Diagnosis']


training_set, test_set, class_set, test_class_set = train_test_split(space,
                                                                    predict_class,
                                                                    test_size = 0.20, 
                                                                    random_state = 50)


# In[4]:


fit_random_forest = RandomForestClassifier(random_state=42)


# In[5]:


np.random.seed(50)
start = time.time()

param_dist = {'max_depth': [2, 3, 4],
              'bootstrap': [True, False],
              'max_features': ['auto', 'sqrt', 'log2', None],
              'criterion': ['gini', 'entropy']}

cv_rf = GridSearchCV(fit_random_forest, cv = 10,
                     param_grid=param_dist, 
                     n_jobs = 3)

cv_rf.fit(training_set, class_set)
print('Best Parameters using grid search: \n', 
      cv_rf.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))


# In[6]:


fit_random_forest.set_params(criterion = 'entropy',
                  max_features = 'auto', 
                  max_depth = 4)


# In[7]:


fit_random_forest.set_params(warm_start=True, 
                  oob_score=True)

min_estimators = 15
max_estimators = 1000

error_rate = {}
for i in range(min_estimators, max_estimators + 1):
    fit_random_forest.set_params(n_estimators=i)
    fit_random_forest.fit(training_set, class_set)

    oob_error = 1 - fit_random_forest.oob_score_
    error_rate[i] = oob_error


# In[8]:


oob_series = pd.Series(error_rate)


# In[9]:


fig, ax = plt.subplots(figsize=(10, 10))

ax.set_facecolor('#123454')


oob_series.plot(kind='line',
                color = 'red')
plt.axhline(0.055, 
            color='#875FDB',
           linestyle='--')
plt.axhline(0.05, 
            color='#875FDB',
           linestyle='--')
plt.xlabel('n_estimators')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error Rate Across various Forest sizes \n(From 15 to 1000 trees)')


# In[12]:


print('OOB Error rate for a number of 20 trees is: {0:.5f}'.format(oob_series[20]))


# In[13]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
randomforestclassifier = RandomForestClassifier()
randomforestclassifier.fit(training_set,class_set)
rfc_predict = randomforestclassifier.predict(test_set)


# In[14]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(test_class_set, rfc_predict))


# In[23]:


mymodel = RandomForestClassifier(max_depth = 3, n_estimators=10)
mymodel.fit(training_set,class_set)
estimator = mymodel.estimators_[5]
estimator


# In[24]:


from sklearn.tree import export_graphviz
export_graphviz(estimator, out_file='tree_limited.dot', feature_names = ["Age","Smoke","Dia_family_mem","Height","Weight","Pulse","Bp","Date of detection","FBS","ABF","T.Chol","TG","SGPT","Creatinine"],
                class_names = "Diagnosis",
                rounded = True, proportion = False, precision = 2, filled = True)


# In[25]:


get_ipython().system('dot -Tpng randomforesttree.dot -o randomforesttree.png -Gdpi=600')


# In[26]:


from IPython.display import Image
Image(filename = 'tree_limited.png')


# In[ ]:





# In[ ]:




