#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
mydata = pd.read_csv("D:\\445\\13.csv")

mydata.duplicated(subset=None, keep='first')

duplicatevalues = mydata[mydata.duplicated(['Age'])]
print("Duplicate Rows based on a single column are:", duplicatevalues, sep='\n')
duplicatevalues = mydata[mydata.duplicated(['Age', 'Smoke'])]
 
print("Duplicate Rows based on 2 columns are:", duplicatevalues, sep='\n')


# In[ ]:





# In[ ]:




