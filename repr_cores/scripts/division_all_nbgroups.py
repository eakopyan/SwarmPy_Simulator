# coding: utf-8

#============================ IMPORTS ======================================
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from swarm_sim import *


#========================== GLOBAL VARIABLES ==============================
PATH = 'data\\swarm-50-sats-scenario\\coords_v1_if_LLO-'
EXPORT_PATH = 'repr_cores\\output\\data\\'
ROW_DATA = 7

NB_NODES = 50
DURATION = 8641 # Number of data rows, not time!
REVOLUTION = 1800 # Number of data rows
CONNECTION_RANGE = 30 # km

SAMPLE_STEP = 12
NB_REPETITIONS = 20
NB_GROUPS = np.arange(1,NB_NODES+1)

#============================= FUNCTIONS ==================================
def variance(data, mean=None):
    if mean == None:
        mean = np.mean(data)
    return np.mean([(d - mean)**2 for d in data])

def variation_coef(data, mean=None):
    if mean != 0:
        var = variance(data, mean)
        return np.sqrt(var)/mean
    print('Error: mean is null.')
    return -1

def rmse(data, ref=None): # Compare the observed distribution to a reference value
    if ref == None:
        ref = np.mean(data)
    errors = [(e-ref)**2 for e in data]
    ratio = sum(errors)/len(data)
    return np.sqrt(ratio)

#========================== INITIALIZE TOPOLOGY ===========================

satellites = {} # Dict(sat_id: DataFrame)
with tqdm(total=NB_NODES, desc='Extracting data') as pbar:
    for i in range(NB_NODES):
        df_data = pd.read_csv(PATH+str(i)+'.csv', skiprows= lambda x: x<ROW_DATA, header=0)
        satellites[i] = df_data
        pbar.update(1)
        
swarm_data = {} # Dict{timestamp: Swarm}
with tqdm(total=REVOLUTION, desc='Converting to topologies') as pbar:
    for t in range(REVOLUTION):
        swarm_data[t] = Swarm(CONNECTION_RANGE,
                    nodes = [Node(id, sat['xF[km]'].iloc[t], sat['yF[km]'].iloc[t], sat['zF[km]'].iloc[t]) for id,sat in satellites.items()]
                    )
        pbar.update(1)

neighbor_matrices = {} # Dict{timestamp: matrix}
with tqdm(total=REVOLUTION, desc='Computing neighbor matrices') as pbar:
    for t in range(REVOLUTION):
        neighbor_matrices[t] = swarm_data[t].neighbor_matrix()
        pbar.update(1)

topo_graphs = {} # Dict{timestamp: Graph}
with tqdm(total=REVOLUTION, desc='Converting to NetworkX graphs') as pbar:
    for t in range(REVOLUTION):
        topo_graphs[t] = swarm_data[t].swarm_to_nxgraph()
        pbar.update(1)


#============================= REFERENCE METRICS ================================
# Reference temporal evolution: Average Degree, Graph Density, Average Clustering Coefficient
ref_ad, ref_acc = [], []

with tqdm(total=REVOLUTION/SAMPLE_STEP, desc='Reference metrics') as pbar:
    for t in np.arange(0, REVOLUTION, SAMPLE_STEP):
        graph = topo_graphs[t]
        ref_ad.append(np.mean(nx.degree(graph)))
        ref_acc.append(nx.average_clustering(graph))
        pbar.update(1)

#============================== GRAPH DIVISION ==================================

rmse_dict = {
    'Timestamp':[],
    'Nb groups':[],
    'RMSE AD':[],
    'RMSE ACC':[]
}


ALGO = 'RND'
print('\nPerforming graph division:', ALGO, '\t\tNumber of repetitions:', NB_REPETITIONS)

with tqdm(total=len(NB_GROUPS), desc='Number of groups') as group_bar:
    for nb_group in NB_GROUPS:
        for rep in range(NB_REPETITIONS):
            swarm_data[0].reset_groups()
            groups = swarm_data[0].RND(n=nb_group, s=rep+1, by_id=True)# <==================== ALGO CHOICE 
            
            for t in np.arange(0, REVOLUTION, SAMPLE_STEP):
                graph = topo_graphs[t]
                group_ad = [] # List(mean degree of group)
                group_acc = []

                for group_id, node_list in groups.items():
                    if len(node_list)>0:
                        group_ad.append(np.mean(nx.degree(graph, node_list)))
                        group_acc.append(nx.average_clustering(graph, node_list))
                    
                rmse_ad = rmse(group_ad, ref_ad[int(t/SAMPLE_STEP)]) 
                rmse_acc = rmse(group_acc, ref_acc[int(t/SAMPLE_STEP)]) 
                
                rmse_dict['Timestamp'].append(t)
                rmse_dict['Nb groups'].append(nb_group)
                rmse_dict['RMSE AD'].append(rmse_ad)
                rmse_dict['RMSE ACC'].append(rmse_acc)      
        group_bar.update(1)
            
            
#===================================== EXPORT DATA ===================================        
df = pd.DataFrame(rmse_dict)
filename = 'sat50_RMSE_'+ALGO+'_sampled'+str(SAMPLE_STEP)+'_rep'+str(NB_REPETITIONS)+'.csv'
print('\nExporting to', EXPORT_PATH+filename)
df.to_csv(EXPORT_PATH+filename, sep=',')