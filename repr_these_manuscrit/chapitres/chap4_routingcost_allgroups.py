# coding: utf-8

#============================ IMPORTS ======================================
import numpy as np
import pandas as pd
from tqdm import tqdm
from swarm_sim import *



#========================== GLOBAL VARIABLES ==============================
PATH = 'data\\cnes_swarm50\\track_'
EXPORT_PATH = 'repr_these_manuscrit\\output\\data\\'

NB_NODES = 50
DURATION = 10000   # Nb samples
REVOLUTION = 1800  # Nb samples
SAMPLE_FREQ = 0.1  # Hz, 1 sample every 10 seconds
CONNECTION_RANGE = 30000 # m

SAMPLE_STEP = 12
NB_REPETITIONS = 30
NB_GROUPS = np.arange(1,NB_NODES+1)


#============================= FUNCTIONS ==================================
def routing_cost(swarm, group=None):
    rcost = []
    nodes = swarm.graph.nodes
    if group:
        nodes = group
    for src in nodes:
        for dst in nodes:
            if nx.has_path(swarm.graph, src, dst):
                rcost.append(nx.shortest_path_length(swarm.graph, src, dst, weight='cost'))
    return sum(rcost)


#========================== INITIALIZE TOPOLOGY ===========================
satellites = {} # Dict(sat_id: DataFrame)
with tqdm(total=NB_NODES, desc='Extracting data') as pbar:
    for i in range(NB_NODES):
        df = pd.read_csv(PATH+str(i)+'.csv')
        df['coords'] = ['x','y','z']
        satellites[i] = df.set_index('coords', drop=True)
        pbar.update(1)
        
swarm_data = {} # Dict{timestamp: Swarm}
with tqdm(total = REVOLUTION, desc = 'Converting to Swarm') as pbar:
    for t in range(REVOLUTION):
        swarm_data[t] = Swarm(
            connection_range=CONNECTION_RANGE, 
            nodes=[Node(id, node[str(t)].x, node[str(t)].y, node[str(t)].z) for id,node in satellites.items()]
            )
        pbar.update(1)
        
neighbor_matrices = {} # Dict{timestamp: matrix}
with tqdm(total=REVOLUTION, desc='Computing neighbor matrices') as pbar:
    for t in range(REVOLUTION):
        neighbor_matrices[t] = swarm_data[t].neighbor_matrix(weighted=True)
        pbar.update(1)
        
# Création des graphes associés  
with tqdm(total=REVOLUTION, desc='Generating graphs') as pbar:
    for t in range(REVOLUTION):
        swarm_data[t].create_graph()
        pbar.update(1)
        
# Enlever les ISL trop chers de l'essaim (ceux dont le coût est supérieur au coût du plus court chemin)
with tqdm(total=REVOLUTION, desc = 'Removing expensive edges') as pbar:
    for t in range(REVOLUTION):
        swarm_data[t].remove_expensive_edges()
        pbar.update(1)
        
        
      
#============================== GRAPH DIVISION ==================================
rcost_dict = {
    'Timestamp':[],
    'Nb groups':[],
    'Rcost':[]
}


ALGO = 'FFD'
print('\nDivision algorithm:', ALGO, '\t\tNumber of repetitions:', NB_REPETITIONS)

for nb_group in NB_GROUPS:
    with tqdm(total=NB_REPETITIONS, desc='Group '+str(nb_group)) as group_bar:
        for rep in range(NB_REPETITIONS):
            swarm_data[0].reset_groups()
            groups = swarm_data[0].FFD(n=nb_group, s=rep+1)# <==================== ALGO CHOICE 
            cost_inter = len(groups.keys())*(len(groups.keys())-1)

            for t in np.arange(0, REVOLUTION, SAMPLE_STEP):
                group_rcost = [routing_cost(swarm_data[t], gr) for gr in groups.values() if len(gr)>0]
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