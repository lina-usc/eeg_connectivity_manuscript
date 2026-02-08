#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import random


# In[6]:


fig, axes = plt.subplots(1,3,figsize=(12,3))

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

###### Ground truth ################


edge_list_1 = [('y0','y1',{'w':'A1'}),('y1','y2',{'w':'B1'}),('y0','y2',{'w':'C1'})]

edge_list_2 = [('y1','y0',{'w':'A2'}),('y2','y1',{'w':'B2'}),('y2','y0',{'w':'C2'})]

G = nx.DiGraph()

G.add_edges_from(edge_list_1)
pos=nx.spring_layout(G,seed=5)
pos = nx.shell_layout(G,rotate=np.pi/2)
arc_rad = 0.05


nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=edge_list_1, connectionstyle=f'arc3, rad = {arc_rad}',alpha=[0,0,0],arrowsize=25)
nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=edge_list_2, connectionstyle=f'arc3, rad = {arc_rad}',alpha=[0,1,1],arrowsize=25)
nx.draw_networkx_edge_labels(G, pos, ax=axes[0], edge_labels={('y1','y0'):0,('y2','y1'):1,('y2','y0'):1},font_color='red')
nx.draw_networkx_nodes(G, pos, ax=axes[0],node_color='orange',alpha=1.0)
nx.draw_networkx_labels(G, pos, ax=axes[0])

axes[0].set_title("Common input", fontsize=12, fontweight="bold")

nx.draw_networkx_edges(G, pos, ax=axes[1], edgelist=edge_list_1, connectionstyle=f'arc3, rad = {arc_rad}',alpha=[0,0,0],arrowsize=25)
nx.draw_networkx_edges(G, pos, ax=axes[1], edgelist=edge_list_2, connectionstyle=f'arc3, rad = {arc_rad}',alpha=[1,1,0],arrowsize=25)
nx.draw_networkx_edge_labels(G, pos, ax=axes[1], edge_labels={('y1','y0'):1,('y2','y1'):1,('y2','y0'):0},font_color='red')
nx.draw_networkx_nodes(G, pos, ax=axes[1],node_color='orange',alpha=1.0)
nx.draw_networkx_labels(G, pos, ax=axes[1])

axes[1].set_title("Indirect connections", fontsize=12, fontweight="bold")

nx.draw_networkx_edges(G, pos, ax=axes[2], edgelist=edge_list_1, connectionstyle=f'arc3, rad = {arc_rad}',alpha=[0,0,0],arrowsize=25)
nx.draw_networkx_edges(G, pos, ax=axes[2], edgelist=edge_list_2, connectionstyle=f'arc3, rad = {arc_rad}',alpha=[0,0,1],arrowsize=25)
nx.draw_networkx_edge_labels(G, pos, ax=axes[2], edge_labels={('y1','y0'):0,('y2','y1'):0,('y2','y0'):1},font_color='red')
nx.draw_networkx_nodes(G, pos, ax=axes[2],node_color='orange',alpha=1.0)
nx.draw_networkx_labels(G, pos, ax=axes[2])

axes[2].set_title("Volume conduction", fontsize=12, fontweight="bold")

plt.savefig("/scratch/figures/confounders_fig.png", dpi=300)


# In[ ]:





# In[ ]:




