#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
d = pd.read_csv("D:\\445\\13.csv")
meanvalue = d['Age'].mean()
print ('Mean Age: ' + str(meanvalue))


# In[4]:


import pandas as pd
import numpy as np
df = pd.read_csv("D:\\445\\13.csv")
print (df.isnull().sum())


# In[ ]:




