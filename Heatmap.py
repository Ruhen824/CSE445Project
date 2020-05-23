#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import svm

import itertools

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


import seaborn

get_ipython().run_line_magic('matplotlib', 'inline')
data  = pd.read_csv("D:\\445\\KhidmahActuall.csv")
df = pd.DataFrame(data)
df_corr = df.corr()
plt.figure(figsize=(15,10))
seaborn.heatmap(df_corr, cmap="YlGnBu") # Displaying the Heatmap
seaborn.set(font_scale=2,style='white')

plt.title('Heatmap correlation')
plt.show()


# In[13]:





# In[11]:





# In[ ]:




