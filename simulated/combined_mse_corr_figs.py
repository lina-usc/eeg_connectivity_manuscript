#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

FIGURES_DIR = pathlib.Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# In[14]:


figures = ['mse_func_non-dyn.png', 'mse_func_dyn.png']
margin = 10
fig, axes = plt.subplots(1, 2, figsize=(10, 15))
for figure, ax in zip(figures, axes.ravel()):
    img = mpimg.imread(FIGURES_DIR / figure)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set(frame_on=False)

    ax.imshow(img)

plt.subplots_adjust(wspace=0, hspace=0)

fig.tight_layout(pad=0)
fig.savefig(FIGURES_DIR / "mse_combined_func.png", bbox_inches='tight', dpi=300)


# In[15]:


figures = ['mse_eff_non-dyn.png', 'mse_eff_dyn.png']
margin = 10
fig, axes = plt.subplots(1, 2, figsize=(10, 15))
for figure, ax in zip(figures, axes.ravel()):
    img = mpimg.imread(FIGURES_DIR / figure)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set(frame_on=False)

    ax.imshow(img)

plt.subplots_adjust(wspace=0, hspace=0)

fig.tight_layout(pad=0)
fig.savefig(FIGURES_DIR / "mse_combined_eff.png", bbox_inches='tight', dpi=300)


# In[ ]:




