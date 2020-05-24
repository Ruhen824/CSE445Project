#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')



import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

d = pd.read_csv("D:\\445\\13.csv")

plt.scatter(d['Age'],d['Diagnosis'])

plt.xlabel('Age')

plt.ylabel('Diagnosis')

plt.title('Scatter plot on dataset')

sns.boxplot(x='Diagnosis',y='FBS',data=d)


# In[2]:



sns.violinplot(x='Diagnosis',y='ABF',data=d,size=6)


# In[3]:


sns.violinplot(x='Diagnosis',y='SGPT',data=d,size=6)


# In[4]:



import seaborn as sns

sns.countplot(x='Smoke',data=d,palette='hls')

plt.show()


# In[5]:



get_ipython().run_line_magic('matplotlib', 'inline')

pd.crosstab(d.Age,d.Smoke).plot(kind='bar')

plt.title('How Age effect diagnosis')

plt.xlabel('Age')

plt.ylabel('Smoke')


# In[6]:


table=pd.crosstab(d.FBS,d.Diagnosis)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Stacked Bar Chart ')

plt.xlabel('FBS')

plt.ylabel('Diagnosis')


# In[ ]:




