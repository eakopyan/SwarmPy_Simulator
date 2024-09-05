# coding: utf-8

#============================ IMPORTS ======================================
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from swarm_sim import *



#========================== GLOBAL VARIABLES ==============================
PATH = 'data\\cnes_swarm50\\track_'
EXPORT_PATH = 'repr_these_manuscrit\\output\\data\\'

NB_NODES = 50       # Nombre de satellites (noeuds)
DURATION = 10000    # Nombre total d'échantillons disponibles
REVOLUTION = 1800   # Nombre d'échantillons pour 1 révolution en orbite lunaire
SAMPLE_FREQ = 0.1   # Fréquence d'échantillonnage (Hz) : toutes les 10 secondes
CONNECTION_RANGE = 30000    # Portée de connexion (m)

# Variables globales pour l'analyse de la division
SAMPLE_STEP = 12    # Fréquence de ré-échantillonnage pour ne pas analyser toutes les topologies (1 sur 12, i.e. toutes les 2 minutes)


#============================= FUNCTIONS ==================================
def add_data_row(data_dict,tsp,rflow, rcost, eff, red, disp, crit):
        data_dict['Timestamp'].append(tsp)
        data_dict['RFlow'].append(rflow)
        data_dict['RCost'].append(rcost)
        data_dict['Efficiency'].append(eff)
        data_dict['Redundancy'].append(red)
        data_dict['Disparity'].append(disp)
        data_dict['Criticity'].append(crit)


def swarm_flow_nb():
    return NB_NODES*(NB_NODES-1)/2


def pair_disparity(shortest_paths:list):
    if len(shortest_paths)==1:
        return 0.0
    disparity = 0
    max_elem = len(shortest_paths[0]) - 2 # number of intermediate elements
    pairs = []
    for idx1,p1 in enumerate(shortest_paths):
        for idx2,p2 in enumerate(shortest_paths):
            if idx1 != idx2 and set([idx1,idx2]) not in pairs:
                pairs.append(set([idx1,idx2]))
                common_elem = set(p1).intersection(p2)
                disparity += 1 - (len(common_elem)-2)/max_elem
    return disparity/len(pairs)



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
# Dict to store data (convert later into pd.DataFrame)
final_data = {
    'Timestamp':[],
    'RFlow':[],
    'RCost':[],
    'Efficiency':[],
    'Redundancy':[],
    'Disparity':[],
    'Criticity': []
}

print('\nNo graph division here.\n')

nb_flow = swarm_flow_nb()

with tqdm(total=REVOLUTION/SAMPLE_STEP, desc='Temporal evolution') as pbar:
    for t in np.arange(0, REVOLUTION, SAMPLE_STEP):
        swarm = swarm_data[t]
        graph = swarm.graph

        visited_pairs, paths = [], []
        redundancies = []
        disparities = []
        rcost = 0
        pair_efficiency = 0.0

        for src_id in graph.nodes:
            for dst_id in graph.nodes:
                if dst_id != src_id and set((src_id,dst_id)) not in visited_pairs:  
                    visited_pairs.append(set((src_id,dst_id))) 
                    pair_efficiency += nx.efficiency(graph, src_id, dst_id)
                    if nx.has_path(graph, src_id, dst_id):
                        paths.append(set((src_id,dst_id))) 
                        shortest_paths = list(nx.all_shortest_paths(graph, src_id, dst_id, weight='cost'))
                        spl = len(shortest_paths[0]) - 1
                        rcost += spl
                        redundancies.append(len(shortest_paths))
                        disparities.append(pair_disparity(shortest_paths))

        rflow = len(paths)/nb_flow
        df_bc = pd.DataFrame(swarm.betweeness_centrality())
        crit = len(df_bc[df_bc['BC']>=0.05]['BC'])
        rcost = rcost*2
        swarm_efficiency = pair_efficiency/nb_flow

        add_data_row(final_data,
                        t,
                        rflow,
                        rcost, 
                        swarm_efficiency,  
                        np.mean(redundancies),
                        np.mean(disparities),
                        crit)
        pbar.update(1)
        
        
#===================================== EXPORT DATA ===================================        
results_df = pd.DataFrame(final_data)
print(results_df.head())
print(results_df.shape[0], 'rows')

filename = 'sat50_reliability_undivided_sampled'+str(SAMPLE_STEP)+'.csv'
print('\nExporting to', EXPORT_PATH+filename)
results_df.to_csv(EXPORT_PATH+filename, sep=',')