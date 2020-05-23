#!/usr/bin/env python
# coding: utf-8

# In[6]:



import pandas as pd
DataFrame = pd.read_csv("D:\\445\\13.csv")

DataFrame.duplicated(subset=None, keep='first')


# In[8]:


duplicateRowsDF = DataFrame[DataFrame.duplicated(['Age'])]
print("Duplicate Rows based on a single column are:", duplicateRowsDF, sep='\n')


# In[11]:


duplicateRowsDF = DataFrame[DataFrame.duplicated(['Age', 'Smoke'])]
 
print("Duplicate Rows based on 2 columns are:", duplicateRowsDF, sep='\n')


# In[ ]:




