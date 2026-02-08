#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# In[57]:


corr_dfs = ["corr_df_func.csv", "corr_df_eff.csv"]
all_labels = [['coh','imCoh','cohy','PLV','ciPLV','PPC','PLI','uPLI$^2$', 'wPLI', 'dwPLI$^2$'], 
         ['pSGP', 'gPDC', 'dDTF', 'DTF', 'DC', 'PDC']]


# In[64]:


fig, axes = plt.subplots(1,2,figsize=(10,4))
for df,labels,ax in zip(corr_dfs,all_labels,axes):
    corr_df = pd.read_csv(df)
    corr_df.drop("Unnamed: 0",axis=1,inplace=True)
    corr_df[''] = labels
    corr_df.set_index('', inplace=True)
    corr_df.columns = labels
    sns.heatmap(corr_df,annot=True, fmt= ".2f", annot_kws={"size":8},ax=ax)
fig.tight_layout()
plt.savefig("corr_matrices.png",dpi=300)    
        


# In[ ]:




