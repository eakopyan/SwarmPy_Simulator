# coding: utf-8

#============================ IMPORTS ======================================
import numpy as np
import pandas as pd
import networkx as nx
import os
from tqdm import tqdm
from swarm_sim import *


#========================== GLOBAL VARIABLES ==============================
PATH = 'data\\swarm-50-sats-scenario\\coords_v1_if_LLO-'
EXPORT_PATH = 'repr_adhoc_networks\\output\\data'
ROW_DATA = 7


CONNECTION_RANGE = 30 # km
NB_NODES = 50
DURATION = 8641 # Number of data rows, not time!
REVOLUTION = 1800 # Number of data rows
SAMPLE_STEP = 12 # Take one out of 12 samples (alleviates calculations)


#============================= FUNCTIONS ==================================
   
def swarm_betweeness_centrality(graph, tsp):
    bc = nx.betweenness_centrality(graph)
    bc_dict = {
        'Node':list(bc.keys()),
        'NBC':list(bc.values())
    }
    df_bc = pd.DataFrame(bc_dict)
    df_bc['Timestamp'] = tsp
    return df_bc



#============================ INITIALIZE TOPOLOGY ==============================

satellites = {} # Dict(sat_id: DataFrame)
with tqdm(total=NB_NODES, desc='Extracting data') as pbar:
    for i in range(NB_NODES):
        df_data = pd.read_csv(PATH+str(i)+'.csv', skiprows= lambda x: x<ROW_DATA, header=0)
        satellites[i] = df_data
        pbar.update(1)
        
swarm_data = {} # Dict{timestamp: Swarm}
with tqdm(total=REVOLUTION, desc='Converting to topologies') as pbar:
    for t in range(REVOLUTION):
        swarm_data[t] = Swarm(connection_range=30,
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


#==================================== BEGIN =====================================

# DataFrame to store data
final_df = pd.DataFrame(columns=['Timestamp','Node','NBC'])

print('\nNo graph division here.\n')


with tqdm(total=REVOLUTION/SAMPLE_STEP, desc='Critical nodes distribution') as pbar:
    for t in np.arange(0, REVOLUTION, SAMPLE_STEP):
        graph = topo_graphs[t]
        df_temp = swarm_betweeness_centrality(graph, t)
        final_df = pd.concat([final_df, df_temp], ignore_index=True)

        pbar.update(1)
 

#=============================== EXPORTING RESULTS ============================

print(final_df.head())
print(final_df.shape[0], 'rows')

filename = 'sat50_criticity_undivided_sampled'+str(SAMPLE_STEP)+'.csv'
print('\nExporting to', os.path.join(EXPORT_PATH, filename))
final_df.to_csv(os.path.join(EXPORT_PATH, filename), sep=',')