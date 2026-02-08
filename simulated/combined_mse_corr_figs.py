#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# In[14]:


figures = ['mse_func_non-dyn.png','mse_func_dyn.png']
margin = 10
fig, axes = plt.subplots(1, 2, figsize=(10,15))
for figure, ax in zip(figures, axes.ravel()):
    img = mpimg.imread(f"{figure}")
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set(frame_on=False)
    
    ax.imshow(img)        
    #ax.set_aspect(1)    

#fig.tight_layout(rect=[-0.01, -0.01, .01, .01], w_pad=0.0, h_pad=0.0)
plt.subplots_adjust(wspace=0, hspace=0)

fig.tight_layout(pad=0)
fig.savefig("mse_combined_func.png", bbox_inches='tight',dpi=300)


# In[15]:


figures = ['mse_eff_non-dyn.png','mse_eff_dyn.png']
margin = 10
fig, axes = plt.subplots(1, 2, figsize=(10,15))
for figure, ax in zip(figures, axes.ravel()):
    img = mpimg.imread(f"{figure}")
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set(frame_on=False)
    
    ax.imshow(img)        
    #ax.set_aspect(1)    

#fig.tight_layout(rect=[-0.01, -0.01, .01, .01], w_pad=0.0, h_pad=0.0)
plt.subplots_adjust(wspace=0, hspace=0)

#fig.suptitle('Common input \n', fontsize=10)
fig.tight_layout(pad=0)
fig.savefig("mse_combined_eff.png", bbox_inches='tight',dpi=300)


# In[ ]:




