#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import seaborn as sns
import mne
from mne_connectivity import spectral_connectivity_epochs
from spectral_connectivity import Multitaper, Connectivity
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import time
from matplotlib.patches import FancyArrowPatch
from scipy.stats import f_oneway
#from my_networkx import my_draw_networkx_edge_labels
import random


# ### DEFINING FUNCTIONS

# In[9]:


confounder_list = ['common_input', 'indirect_connections', 'volume_conduction']

def simulate_confounder(confounder):

    #setting parameters of signal
    sfreq = 250 # Hz
    tmin = 0 # s
    tmax = 100 # s
    time = np.arange(tmin, tmax, 1/sfreq) 

    f1 = np.random.randint(1,41)  
    f2 = np.random.randint(1,41)  
    f3 = np.random.randint(1,41)  

    rho1 = round(random.uniform(0.5, 1.0),2) #literature value 0.95
    rho2 = round(random.uniform(0.5, 1.0),2) #literature value 0.8
    rho3 = round(random.uniform(0.5, 1.0),2) #literature value 0.9

    
    #signals of 100s duration, 250 Hz sampling frequency, randomly chosen frequency
    signal1 = np.sin(2*np.pi*f1*time)
    signal2 = np.sin(2*np.pi*f2*time)
    signal3 = np.sin(2*np.pi*f3*time)

    #random white noise time series
    noise1 = np.random.randn(len(time))
    noise2 = np.random.randn(len(time))
    noise3 = np.random.randn(len(time))

    #gains
    g1 = g2 = g3 = 1

    
    #Factor scaling amplitude of noise
    alpha1 = np.random.uniform(0.01, 0.5)
    alpha2 = np.random.uniform(0.01, 0.5)
    alpha3 = np.random.uniform(0.01, 0.5)

    
    #signal to noise ratio
    snr1 = g1/alpha1
    snr2 = g2/alpha2
    snr3 = g3/alpha3

    #phase
    phase_3_1 = np.random.uniform(0,2)*np.pi
    phase_3_2 = np.random.uniform(0,2)*np.pi
    phase_1_2 = np.random.uniform(0,2)*np.pi

    #delays in seconds
    tau3_1 = phase_3_1/(2*np.pi*f1) #from 3 to 1
    tau3_2 = phase_3_2/(2*np.pi*f2) #from 3 to 2
    tau1_2 = phase_1_2/(2*np.pi*f3) #from 1 to 2
    
    #Time lags in samples
    lag3_1 = np.int64(np.round(tau3_1*f1))
    lag3_2 = np.int64(np.round(tau3_2*f2))
    lag1_2 = np.int64(np.round(tau1_2*f3))

    
    #c values
    coefficient_dict = {'common_input':{'c_3_1': 1, 'c_3_2':1, 'c_1_2':0},
                        'indirect_connections':{'c_3_1': 1, 'c_3_2':0, 'c_1_2':1},
                        'volume_conduction':{'c_3_1': 0, 'c_3_2': 0, 'c_1_2':0}}

    c_3_1 = coefficient_dict[confounder]['c_3_1']
    c_3_2 = coefficient_dict[confounder]['c_3_2']
    c_1_2 = coefficient_dict[confounder]['c_1_2']
    
    y1 = np.zeros(len(time))
    y2 = np.zeros(len(time))
    y3 = np.zeros(len(time))

    for i in np.arange(2, len(time)):
        y3[i] = rho3*y3[i-1]-((rho3**2)*y3[i-2])+noise3[i]*snr3
        y1[i] = rho1*y1[i-1]-((rho1**2)*y1[i-2])- ((c_3_1)*y3[i-1]) +noise1[i]*snr1
        y2[i] = rho2*y2[i-1]-((rho2**2)*(y2[i-2]))- (c_1_2*(y1[i-3])) -(c_3_2*(y3[i-2]))+noise2[i]*snr2

    if confounder == 'volume_conduction':
        L = np.random.uniform(-1,1,size=(3,3))
    else:
        L = np.identity(3)
    

    z1, z2, z3 = L @ np.array([y1, y2, y3])

    return[z1, z2, z3]


# In[2]:


def simulated_raw(signals):
    sim_data = np.array(signals)
    info = mne.create_info(ch_names=["y1", "y2", "y3"], ch_types=["eeg"]*3, sfreq=250)
    simulated_raw = mne.io.RawArray(sim_data, info)
    return simulated_raw


# In[3]:


def estimate_conn_undirected(method, data):
    epochs_1 = mne.make_fixed_length_epochs(simulated_raw(data), duration=1)
    
    mne_con = spectral_connectivity_epochs(epochs_1, method=method, sfreq=250, fmin=8, fmax=13, 
                                           fskip=0, faverage=False,
                                           mt_low_bias=True, block_size=1000, n_jobs=1, 
                                           verbose=None)
    con_mat = mne_con.get_data(output="dense")
    con_mat_normalized = (con_mat - con_mat.min())/(con_mat.max() - con_mat.min())
    return con_mat_normalized


# In[4]:


def estimate_conn_directed(method, data):
    epochs_2 = mne.make_fixed_length_epochs(simulated_raw(data), duration=8)
    
    m = Multitaper(time_series=np.array(epochs_2.get_data()).transpose(2, 0, 1), sampling_frequency=250)
    c = Connectivity(fourier_coefficients=m.fft(), frequencies=m.frequencies, time=m.time)    
    con = getattr(c, method)()
    con_mat = np.nan_to_num(con.squeeze())[80:131].transpose(1,2,0)
    con_mat_normalized = (con_mat - con_mat.min())/(con_mat.max() - con_mat.min())
    return con_mat_normalized


# ### DEFINING GROUND TRUTHS AND METHODS

# In[7]:


ground_truth_dict = {'common_input':np.array([[0,0,0],
                                                 [0,0,0],
                                                 [1,1,0]]),
                     'indirect_connections':np.array([[0,0,0],
                                            [1,0,0],
                                            [1,0,0]]),
                     'volume_conduction':np.array([[0,0,0],
                                            [0,0,0],
                                            [0,0,0]])}

undirected_methods = ['coh', 'ciplv','imcoh','wpli2_debiased']

directed_methods = ['generalized_partial_directed_coherence', 
                    'direct_directed_transfer_function', 
                    'pairwise_spectral_granger_prediction']


# ### TOPOLOGY ESTIMATES AND GRAPHS

# ### Undirected methods

# In[11]:


get_ipython().run_cell_magic('capture', '', 'edge_lists_undirected = {}\nfor confounder in confounder_list:\n    edge_lists = {}\n    for method in undirected_methods:\n        edges_list = []\n        for i in range(100):\n            con_mat = estimate_conn_undirected(method,simulate_confounder(confounder))\n            edges = [round(con_mat.mean(2)[1][0],2), round(con_mat.mean(2)[2][1],2), round(con_mat.mean(2)[2][0],2)]\n            edges_list.append(edges)\n        mean_edges = np.array(edges_list)\n        edge_lists[method] = np.round(mean_edges.mean(axis=0),2)\n    edge_lists_undirected[confounder] = edge_lists\n')


# In[12]:


edge_lists_undirected


# In[13]:


edge_list_1 = [('X','Y',{'w':'A1'}),('Y','Z',{'w':'B1'}),('X','Z',{'w':'C1'})]

edge_list_2 = [('Y','X',{'w':'A2'}),('Z','Y',{'w':'B2'}),('Z','X',{'w':'C2'})]


# #### Common input

# In[14]:


fig, axes = plt.subplots(5,1,figsize=(3,8))

G = nx.DiGraph()

G.add_edges_from(edge_list_1)
pos=nx.spring_layout(G,seed=5)
pos = nx.shell_layout(G,rotate=np.pi/2)
arc_rad = 0.05
nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=edge_list_1, connectionstyle=f'arc3, rad = {arc_rad}',alpha=[0,0,0],arrowsize=25)
nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=edge_list_2, connectionstyle=f'arc3, rad = {arc_rad}',alpha=[0,1,1],arrowsize=25)
nx.draw_networkx_edge_labels(G, pos, ax=axes[0], edge_labels={('Y','X'):0,('Z','Y'):1,('Z','X'):1},font_color='red')
nx.draw_networkx_nodes(G, pos, ax=axes[0],node_color='orange',alpha=1.0)
nx.draw_networkx_labels(G, pos, ax=axes[0])
#axes[0].set_title('Common input confounder',fontsize=11)

for ax,method in zip(list(range(1,5)),undirected_methods):
    G = nx.DiGraph()
    edge_list = [('X','Y',{'w':'A1'}),
                 ('Y','Z',{'w':'B1'}),
                 ('X','Z',{'w':'C1'})]
    G.add_edges_from(edge_list)
    pos = nx.spring_layout(G,seed=5)
    pos = nx.shell_layout(G, rotate=np.pi/2, scale=0.1)
    
    alpha = edge_lists_undirected['common_input'][method]
    nx.draw_networkx_edges(G, pos, ax=axes[ax], edgelist=edge_list, alpha=alpha,arrowstyle='<->')
    nx.draw_networkx_edge_labels(G, pos, ax=axes[ax], edge_labels={('X','Y'):alpha[0],('Y','Z'):alpha[1],('X','Z'):alpha[2]},font_color='red')
    arc_rad = 0.05
    nx.draw_networkx_nodes(G, pos, ax=axes[ax], node_color='grey',alpha=1.0)
    nx.draw_networkx_labels(G, pos, ax=axes[ax])

axes[0].set_ylabel(('Ground truth'), rotation='horizontal',labelpad=50)
axes[1].set_ylabel(('coh'), rotation='horizontal',labelpad=50)
axes[2].set_ylabel(('ciPLV'), rotation='horizontal',labelpad=50)
axes[3].set_ylabel(('imcoh'), rotation='horizontal',labelpad=50)
axes[4].set_ylabel(('dwPLI'), rotation='horizontal',labelpad=50)

fig.suptitle('                   Non-dynamic system')
fig.tight_layout()
plt.savefig('sim_topo_func_comm_dyn.png',dpi=300)


# #### Indirect connections

# In[15]:


fig, axes = plt.subplots(5,1,figsize=(3,8))

G = nx.DiGraph()

G.add_edges_from(edge_list_1)
pos=nx.spring_layout(G,seed=5)
pos = nx.shell_layout(G,rotate=np.pi/2)
arc_rad = 0.05
nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=edge_list_1, connectionstyle=f'arc3, rad = {arc_rad}',alpha=[0,0,0],arrowsize=25)
nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=edge_list_2, connectionstyle=f'arc3, rad = {arc_rad}',alpha=[0,1,1],arrowsize=25)
nx.draw_networkx_edge_labels(G, pos, ax=axes[0], edge_labels={('Y','X'):0,('Z','Y'):1,('Z','X'):1},font_color='red')
nx.draw_networkx_nodes(G, pos, ax=axes[0],node_color='orange',alpha=1.0)
nx.draw_networkx_labels(G, pos, ax=axes[0])
#axes[0].set_title('Common input confounder',fontsize=11)

for ax,method in zip(list(range(1,5)),undirected_methods):
    G = nx.DiGraph()
    edge_list = [('X','Y',{'w':'A1'}),
                 ('Y','Z',{'w':'B1'}),
                 ('X','Z',{'w':'C1'})]
    G.add_edges_from(edge_list)
    pos = nx.spring_layout(G,seed=5)
    pos = nx.shell_layout(G, rotate=np.pi/2, scale=0.1)
    
    alpha = edge_lists_undirected['indirect_connections'][method]
    nx.draw_networkx_edges(G, pos, ax=axes[ax], edgelist=edge_list, alpha=alpha,arrowstyle='<->')
    nx.draw_networkx_edge_labels(G, pos, ax=axes[ax], edge_labels={('X','Y'):alpha[0],('Y','Z'):alpha[1],('X','Z'):alpha[2]},font_color='red')
    arc_rad = 0.05
    nx.draw_networkx_nodes(G, pos, ax=axes[ax], node_color='grey',alpha=1.0)
    nx.draw_networkx_labels(G, pos, ax=axes[ax])

axes[0].set_ylabel(('Ground truth'), rotation='horizontal',labelpad=50)
axes[1].set_ylabel(('coh'), rotation='horizontal',labelpad=50)
axes[2].set_ylabel(('ciPLV'), rotation='horizontal',labelpad=50)
axes[3].set_ylabel(('imcoh'), rotation='horizontal',labelpad=50)
axes[4].set_ylabel(('dwPLI'), rotation='horizontal',labelpad=50)

fig.suptitle('                   Non-dynamic system')
fig.tight_layout()
plt.savefig('sim_topo_func_ind_dyn.png',dpi=300)


# #### Volume conduction

# In[16]:


fig, axes = plt.subplots(5,1,figsize=(3,8))

G = nx.Graph()
edge_list = [('X','Y',{'w':'A1'}),
             ('Y','Z',{'w':'B1'}),
             ('X','Z',{'w':'C1'})]
G.add_edges_from(edge_list)
pos = nx.spring_layout(G,seed=5)
pos = nx.shell_layout(G, rotate=np.pi/2, scale=0.1)
nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=edge_list, alpha=[0,0,0])
arc_rad = 0.05
nx.draw_networkx_edge_labels(G, pos, ax=axes[0], edge_labels={('Y','X'):0,('Z','Y'):0,('Z','X'):0},font_color='red')
nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color='orange',alpha=1.0)
nx.draw_networkx_labels(G, pos, ax=axes[0])


for ax,method in zip(list(range(1,5)),undirected_methods):
    G = nx.DiGraph()
    edge_list = [('X','Y',{'w':'A1'}),
                 ('Y','Z',{'w':'B1'}),
                 ('X','Z',{'w':'C1'})]
    G.add_edges_from(edge_list)
    pos = nx.spring_layout(G,seed=5)
    pos = nx.shell_layout(G, rotate=np.pi/2, scale=0.1)
    
    alpha = edge_lists_undirected['volume_conduction'][method]
    nx.draw_networkx_edges(G, pos, ax=axes[ax], edgelist=edge_list, alpha=alpha,arrowstyle='<->')
    nx.draw_networkx_edge_labels(G, pos, ax=axes[ax], edge_labels={('X','Y'):alpha[0],('Y','Z'):alpha[1],('X','Z'):alpha[2]},font_color='red')
    arc_rad = 0.05
    nx.draw_networkx_nodes(G, pos, ax=axes[ax], node_color='grey',alpha=1.0)
    nx.draw_networkx_labels(G, pos, ax=axes[ax])

axes[0].set_title('Non-dynamic system')
axes[0].set_ylabel(('Ground truth'), rotation='horizontal',labelpad=50)
axes[1].set_ylabel(('coh'), rotation='horizontal',labelpad=50)
axes[2].set_ylabel(('ciPLV'), rotation='horizontal',labelpad=50)
axes[3].set_ylabel(('imcoh'), rotation='horizontal',labelpad=50)
axes[4].set_ylabel(('dwPLI'), rotation='horizontal',labelpad=50)

fig.tight_layout()
plt.savefig('sim_topo_func_vol_dyn.png',dpi=300)


# ### Directed methods

# In[17]:


get_ipython().run_cell_magic('capture', '', '\nedge_lists_directed = {}\nfor confounder in confounder_list:\n    edge_lists = {}\n    for method in directed_methods:\n        edges_list = []\n        for i in range(100):\n            con_mat = estimate_conn_directed(method,simulate_confounder(confounder))\n            edges = [round((con_mat.mean(2)[1][0]),2), round((con_mat.mean(2)[2][1]),2), round((con_mat.mean(2)[2][0]),2),round((con_mat.mean(2)[0][1]),2),round((con_mat.mean(2)[1][2]),2), round((con_mat.mean(2)[0][2]),2)]\n            edges_list.append(edges)\n        mean_edges = np.array(edges_list)\n        edge_lists[method] = np.round(mean_edges.mean(axis=0),2)\n        \n    edge_lists_directed[confounder] = edge_lists\n')


# #### Common input

# In[21]:


fig, axes = plt.subplots(4,1,figsize=(5,12))

G = nx.DiGraph()

G.add_edges_from(edge_list_1)

pos = nx.shell_layout(G,rotate=np.pi/2)
arc_rad = 0.05
nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=edge_list_1, connectionstyle=f'arc3, rad = {arc_rad}',alpha=[0,0,0],arrowsize=25)
nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=edge_list_2, connectionstyle=f'arc3, rad = {arc_rad}',alpha=[0,1,1],arrowsize=25)
nx.draw_networkx_edge_labels(G, pos, ax=axes[0], edge_labels={('Y','X'):0,('Z','Y'):1,('Z','X'):1},font_color='red')
nx.draw_networkx_nodes(G, pos, ax=axes[0],node_color='orange',alpha=1.0,node_size=450)
nx.draw_networkx_labels(G, pos, ax=axes[0])


for ax,method in zip(list(range(1,4)),directed_methods):
    alpha = edge_lists_directed['common_input'][method]
    G = nx.DiGraph()
    edge_list_1 = [('X','Y',{'w':'A1'}),('Y','Z',{'w':'B1'}),('X','Z',{'w':'C1'})]
    edge_list_2 = [('Y','X',{'w':'A2'}),('Z','Y',{'w':'B2'}),('Z','X',{'w':'C2'})]
    
    G.add_edges_from(edge_list_1)
    G.add_edges_from(edge_list_2)
    
    pos1 = nx.shell_layout(G,rotate=np.pi/2)
    pos2 = nx.shell_layout(G,rotate=np.pi/2,scale=1.4)

    arc_rad = 0.2
    curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    nx.draw_networkx_edges(G, pos1, ax=axes[ax], edgelist=curved_edges,alpha=[alpha[0],alpha[1],alpha[2],alpha[3],alpha[4],alpha[5]],connectionstyle=f'arc3, rad = {arc_rad}')

    #nx.draw_networkx_edge_labels(G, pos2, ax=axes[ax], edge_labels={('X','Y'):alpha[0],('Y','Z'):alpha[1],('X','Z'):alpha[2]}, font_size=8, font_color='red')
    #nx.draw_networkx_edge_labels(G, pos1, ax=axes[ax], edge_labels={('Y','X'):alpha[3],('Z','Y'):alpha[4],('Z','X'):alpha[5]},font_size=8, font_color='red')

    x1, y1 = pos1['X']
    x2, y2 = pos1['Y']
    axes[ax].text((x1+x2)/2 + 0.08, (y1+y2)/2, alpha[3], fontsize=10, color='red', rotation=60)

    x1, y1 = pos2['Y']
    x2, y2 = pos2['X']
    axes[ax].text((x1+x2)/2 - 0.05, (y1+y2)/2-0.2, alpha[0], fontsize=10, color='red',rotation=60)

    x1, y1 = pos1['Y']
    x2, y2 = pos1['Z']
    axes[ax].text((x1+x2)/2, (y1+y2)/2 + 0.08, alpha[4], fontsize=10, color='red')

    x1, y1 = pos2['Z']
    x2, y2 = pos2['Y']
    axes[ax].text((x1+x2)/2, (y1+y2)/2+0.04, alpha[1], fontsize=10, color='red')

    x1, y1 = pos1['X']
    x2, y2 = pos1['Z']
    axes[ax].text((x1+x2)/2-0.25, (y1+y2)/2, alpha[5], fontsize=10, color='red',rotation=-60)

    x1, y1 = pos2['Z']
    x2, y2 = pos2['X']
    axes[ax].text((x1+x2)/2-0.15, (y1+y2)/2-0.2, alpha[2], fontsize=10, color='red',rotation=-60)

    
    nx.draw_networkx_nodes(G, pos1, ax=axes[ax],node_color='grey',node_size=450)
    nx.draw_networkx_labels(G, pos1, ax=axes[ax])
    
axes[0].set_ylabel(('Ground truth'), rotation='horizontal', fontsize=12, labelpad=50)
axes[1].set_ylabel(('gPDC'), rotation='horizontal',fontsize=12, labelpad=50)
axes[2].set_ylabel(('dDTF'), rotation='horizontal',fontsize=12, labelpad=50)
axes[3].set_ylabel(('pSGP'), rotation='horizontal',fontsize=12,labelpad=50)
    

fig.suptitle('                 Non-dynamic system', fontsize=14)
fig.tight_layout()
plt.savefig('sim_topo_eff_comm_dyn.png')


# #### Indirect connections

# In[19]:


fig, axes = plt.subplots(3,1,figsize=(4.5,10))

G = nx.DiGraph()
edge_list_1 = [('X','Y',{'w':'A1'}),('Y','Z',{'w':'B1'}),('X','Z',{'w':'C1'})]

edge_list_2 = [('Y','X',{'w':'A2'}),('Z','Y',{'w':'B2'}),('Z','X',{'w':'C2'})]

G.add_edges_from(edge_list_1)
G.add_edges_from(edge_list_2)
pos=nx.spring_layout(G,seed=5)
pos = nx.shell_layout(G,rotate=np.pi/2)
arc_rad = 0.05
nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=edge_list_1, connectionstyle=f'arc3, rad = {arc_rad}',alpha=[1,0,0],arrowsize=25)
nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=edge_list_2, connectionstyle=f'arc3, rad = {arc_rad}',alpha=[0,0,1],arrowsize=25)
nx.draw_networkx_edge_labels(G, pos, ax=axes[0], edge_labels={('Y','X'):1,('Z','Y'):0,('Z','X'):1},font_color='red')
nx.draw_networkx_nodes(G, pos, ax=axes[0],node_color='orange',alpha=1.0,node_size=500)
nx.draw_networkx_labels(G, pos, ax=axes[0])

for ax,method in zip(list(range(1,4)),['generalized_partial_directed_coherence','direct_directed_transfer_function']):
    alpha = edge_lists_directed['indirect_connections'][method]
    G = nx.DiGraph()
    edge_list_1 = [('X','Y',{'w':'A1'}),('Y','Z',{'w':'B1'}),('X','Z',{'w':'C1'})]
    edge_list_2 = [('Y','X',{'w':'A2'}),('Z','Y',{'w':'B2'}),('Z','X',{'w':'C2'})]
    
    G.add_edges_from(edge_list_1)
    G.add_edges_from(edge_list_2)
    
    pos1 = nx.shell_layout(G,rotate=np.pi/2)
    pos2 = nx.shell_layout(G,rotate=np.pi/2,scale=1.0)

    arc_rad = 0.2
    curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    nx.draw_networkx_edges(G, pos1, ax=axes[ax], edgelist=edge_list_1,alpha=[alpha[0],alpha[1],alpha[2]],connectionstyle=f'arc3, rad = {arc_rad}')
    nx.draw_networkx_edges(G, pos2, ax=axes[ax], edgelist=edge_list_2,alpha=[alpha[3],alpha[4],alpha[5]],connectionstyle=f'arc3, rad = {arc_rad}')

    #nx.draw_networkx_edge_labels(G, pos1, ax=axes[ax], edge_labels={('X','Y'):alpha[0],('Y','Z'):alpha[1],('X','Z'):alpha[2]}, font_size=8, font_color='red')
    #nx.draw_networkx_edge_labels(G, pos2, ax=axes[ax], edge_labels={('Y','X'):alpha[3],('Z','Y'):alpha[4],('Z','X'):alpha[5]}, font_size=8, font_color='red')
    
    nx.draw_networkx_nodes(G, pos1, ax=axes[ax],node_color='grey',node_size=500)
    nx.draw_networkx_labels(G, pos1, ax=axes[ax])

    x1, y1 = pos1['X']
    x2, y2 = pos1['Y']
    axes[ax].text((x1+x2)/2, (y1+y2)/2-0.15, alpha[3], fontsize=10, color='red', rotation=60)

    x1, y1 = pos2['Y']
    x2, y2 = pos2['X']
    axes[ax].text((x1+x2)/2-0.25, (y1+y2)/2-0.2, alpha[0], fontsize=10, color='red',rotation=60)

    x1, y1 = pos1['Y']
    x2, y2 = pos1['Z']
    axes[ax].text((x1+x2)/2, (y1+y2)/2 + 0.1, alpha[4], fontsize=10, color='red')

    x1, y1 = pos2['Z']
    x2, y2 = pos2['Y']
    axes[ax].text((x1+x2)/2, (y1+y2)/2 - 0.15, alpha[1], fontsize=10, color='red')

    x1, y1 = pos1['X']
    x2, y2 = pos1['Z']
    axes[ax].text((x1+x2)/2+0.1, (y1+y2)/2-0.2, alpha[5], fontsize=10, color='red',rotation=-60)

    x1, y1 = pos2['Z']
    x2, y2 = pos2['X']
    axes[ax].text((x1+x2)/2-0.15, (y1+y2)/2-0.2, alpha[2], fontsize=10, color='red',rotation=-60)
    
axes[0].set_ylabel(('Ground truth'), rotation='horizontal',labelpad=50,fontsize=12)
axes[0].set_title('Non-dynamic system',fontsize=16)
axes[1].set_ylabel(('gPDC'), rotation='horizontal',labelpad=50,fontsize=12)
axes[2].set_ylabel(('dDTF'), rotation='horizontal',labelpad=50,fontsize=12)
    

#fig.suptitle('                 Non-dynamic system')
fig.tight_layout()
plt.savefig('sim_topo_eff_ind_dyn.png',dpi=300)


# #### Volume conduction

# In[20]:


fig, axes = plt.subplots(4,1,figsize=(5,12))

G = nx.Graph()
edge_list = [('X','Y',{'w':'A1'}),
             ('Y','Z',{'w':'B1'}),
             ('X','Z',{'w':'C1'})]
G.add_edges_from(edge_list)
pos = nx.spring_layout(G,seed=5)
pos = nx.shell_layout(G, rotate=np.pi/2, scale=0.1)
nx.draw_networkx_edges(G, pos, ax=axes[0], edgelist=edge_list, alpha=[0,0,0])
arc_rad = 0.05
nx.draw_networkx_edge_labels(G, pos, ax=axes[0], edge_labels={('Y','X'):0,('Z','Y'):0,('Z','X'):0},font_color='red')
nx.draw_networkx_nodes(G, pos, ax=axes[0], node_color='orange',alpha=1.0, node_size=450)
nx.draw_networkx_labels(G, pos, ax=axes[0])

for ax,method in zip(list(range(1,4)),directed_methods):
    alpha = edge_lists_directed['volume_conduction'][method]
    G = nx.DiGraph()
    edge_list_1 = [('X','Y',{'w':'A1'}),('Y','Z',{'w':'B1'}),('X','Z',{'w':'C1'})]
    edge_list_2 = [('Y','X',{'w':'A2'}),('Z','Y',{'w':'B2'}),('Z','X',{'w':'C2'})]
    
    G.add_edges_from(edge_list_1)
    G.add_edges_from(edge_list_2)
    
    pos1 = nx.shell_layout(G,rotate=np.pi/2)
    pos2 = nx.shell_layout(G,rotate=np.pi/2,scale=1.4)

    arc_rad = 0.2
    curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    nx.draw_networkx_edges(G, pos1, ax=axes[ax], edgelist=curved_edges,alpha=[alpha[0],alpha[1],alpha[2],alpha[3],alpha[4],alpha[5]],connectionstyle=f'arc3, rad = {arc_rad}')

    #nx.draw_networkx_edge_labels(G, pos2, ax=axes[ax], edge_labels={('X','Y'):alpha[0],('Y','Z'):alpha[1],('X','Z'):alpha[2]}, font_size=8, font_color='red')
    #nx.draw_networkx_edge_labels(G, pos1, ax=axes[ax], edge_labels={('Y','X'):alpha[3],('Z','Y'):alpha[4],('Z','X'):alpha[5]},font_size=8, font_color='red')

    x1, y1 = pos1['X']
    x2, y2 = pos1['Y']
    axes[ax].text((x1+x2)/2 + 0.08, (y1+y2)/2, alpha[3], fontsize=10, color='red', rotation=60)

    x1, y1 = pos2['Y']
    x2, y2 = pos2['X']
    axes[ax].text((x1+x2)/2 - 0.05, (y1+y2)/2-0.2, alpha[0], fontsize=10, color='red',rotation=60)

    x1, y1 = pos1['Y']
    x2, y2 = pos1['Z']
    axes[ax].text((x1+x2)/2, (y1+y2)/2 + 0.08, alpha[4], fontsize=10, color='red')

    x1, y1 = pos2['Z']
    x2, y2 = pos2['Y']
    axes[ax].text((x1+x2)/2, (y1+y2)/2+0.04, alpha[1], fontsize=10, color='red')

    x1, y1 = pos1['X']
    x2, y2 = pos1['Z']
    axes[ax].text((x1+x2)/2-0.25, (y1+y2)/2, alpha[5], fontsize=10, color='red',rotation=-60)

    x1, y1 = pos2['Z']
    x2, y2 = pos2['X']
    axes[ax].text((x1+x2)/2-0.15, (y1+y2)/2-0.2, alpha[2], fontsize=10, color='red',rotation=-60)

    nx.draw_networkx_nodes(G, pos1, ax=axes[ax],node_color='grey',node_size=450)
    nx.draw_networkx_labels(G, pos1, ax=axes[ax])
    
axes[0].set_ylabel(('Ground truth'), rotation='horizontal', fontsize=12, labelpad=50)
axes[1].set_ylabel(('gPDC'), rotation='horizontal',fontsize=12, labelpad=50)
axes[2].set_ylabel(('dDTF'), rotation='horizontal',fontsize=12, labelpad=50)
axes[3].set_ylabel(('pSGP'), rotation='horizontal',fontsize=12,labelpad=50)
    

fig.suptitle('                 Non-dynamic system',fontsize=14)
fig.tight_layout()
plt.savefig('sim_topo_eff_vol_dyn.png')


# In[ ]:




