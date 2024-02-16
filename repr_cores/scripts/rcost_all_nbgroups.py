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
NB_REPETITIONS = 30
NB_GROUPS = np.arange(1,NB_NODES+1)

#============================= FUNCTIONS ==================================
def routing_cost(graph, group=None):
    cost = []
    nodes = graph.nodes
    if group:
        nodes = group
    for src in nodes:
        for dst in nodes:
            if nx.has_path(graph, src, dst):
                cost.append(nx.shortest_path_length(graph, src, dst))
    return sum(cost)

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



#============================== GRAPH DIVISION ==================================

rcost_dict = {
    'Timestamp':[],
    'Nb groups':[],
    'Rcost':[]
}


ALGO = 'FFD'
print('\nPerforming graph division:', ALGO, '\t\tNumber of repetitions:', NB_REPETITIONS)

with tqdm(total=len(NB_GROUPS)*NB_REPETITIONS, desc='Groups x rep') as group_bar:
    for nb_group in NB_GROUPS:
        for rep in range(NB_REPETITIONS):
            swarm_data[0].reset_groups()
            groups = swarm_data[0].FFD (n=nb_group, s=rep+1, by_id=True)# <==================== ALGO CHOICE 
            cost_inter = len(groups.keys())*(len(groups.keys())-1)
            
            for t in np.arange(0, REVOLUTION, SAMPLE_STEP):
                graph = topo_graphs[t]
                group_rcost = [routing_cost(graph, gr) for gr in groups.values() if len(gr)>0]
                rcost = sum(group_rcost)+cost_inter
                    
                rcost_dict['Timestamp'].append(t)
                rcost_dict['Nb groups'].append(nb_group)
                rcost_dict['Rcost'].append(rcost)
            group_bar.update(1)
            
            
#===================================== EXPORT DATA ===================================        
df = pd.DataFrame(rcost_dict)
filename = 'sat50_RCOST_'+ALGO+'_sampled'+str(SAMPLE_STEP)+'_rep'+str(NB_REPETITIONS)+'.csv'
print('\nExporting to', EXPORT_PATH+filename)
df.to_csv(EXPORT_PATH+filename, sep=',')