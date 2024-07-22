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

NB_NODES = 50
DURATION = 10000   # Nb samples
REVOLUTION = 1800  # Nb samples
SAMPLE_FREQ = 0.1  # Hz, 1 sample every 10 seconds
CONNECTION_RANGE = 30000 # m

SAMPLE_STEP = 12
NB_REPETITIONS = 30
NB_GROUPS = 10


#============================= FUNCTIONS ==================================
def add_data_row(data_dict,tsp,rflow, rcost, eff, red, disp, crit):
        data_dict['Timestamp'].append(tsp)
        data_dict['RFlow'].append(rflow)
        data_dict['RCost'].append(rcost)
        data_dict['Efficiency'].append(eff)
        data_dict['Redundancy'].append(red)
        data_dict['Disparity'].append(disp)
        data_dict['Criticity'].append(crit)


def swarm_flow_nb(groups):
    nb_pairs = 0
    for group in groups.values():
        group_size = len(group)
        nb_pairs += group_size*(group_size-1)/2
    return nb_pairs


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


def group_betweeness_centrality(graph, sp_all, nb_flow):
    shortest_paths = sp_all['Shortest paths']
    for paths in shortest_paths:
        for path in paths:
            del path[0] # Discard endpoints
            del path[-1]
    bc_dict = {
        'Node':[],
        'BC':[]
    }
    for node in graph.nodes:
        bc = 0
        for paths in shortest_paths:
            sp_node = [path for path in paths if node in path]
            bc += len(sp_node)/len(paths)
        bc = bc/nb_flow #Normalize over all possible pairs
        bc_dict['Node'].append(node)
        bc_dict['BC'].append(bc)
    return bc_dict


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


ALGO = 'FFD'
print('\nDivision algorithm:', ALGO, '\t\tNumber of repetitions:', NB_REPETITIONS)

nb_flow = swarm_flow_nb()

for rep in range(NB_REPETITIONS):
    swarm_data[0].reset_groups()
    groups = swarm_data[0].FFD(n=NB_GROUPS, s=rep) # <==================== ALGO CHOICE 
    nb_flow = swarm_flow_nb(groups)

    with tqdm(total=REVOLUTION/SAMPLE_STEP, desc='Temporal evolution '+str(rep)) as pbar:
        for t in np.arange(0, REVOLUTION, SAMPLE_STEP):
            swarm = swarm_data[t]
            graph = swarm.graph

            sp_all = {
                'Group':[],
                'Source':[],
                'Dest':[],
                'Shortest paths':[]
            }

            visited_pairs, paths = [], []
            redundancies = []
            disparities = []
            rcost = 0
            pair_efficiency = 0.0

            for group_id, group_nodes in groups.items():
                for src_id in group_nodes:
                    for dst_id in group_nodes:
                        if dst_id != src_id and set((src_id,dst_id)) not in visited_pairs:  
                            visited_pairs.append(set((src_id,dst_id))) 
                            pair_efficiency += nx.efficiency(graph, src_id, dst_id)
                            if nx.has_path(graph, src_id, dst_id):
                                paths.append(set((src_id,dst_id))) 
                                shortest_paths = list(nx.all_shortest_paths(graph, src_id, dst_id, weight='cost'))
                                sp_all['Group'].append(group_id)
                                sp_all['Source'].append(src_id)
                                sp_all['Dest'].append(dst_id)
                                sp_all['Shortest paths'].append(shortest_paths)

                                rcost += len(shortest_paths[0]) - 1
                                redundancies.append(len(shortest_paths))
                                disparities.append(pair_disparity(shortest_paths))

            rflow = len(paths)/nb_flow
            df_group = pd.DataFrame(group_betweeness_centrality(graph, sp_all, nb_flow))
            crit = len(df_group[df_group['BC']>=0.05]['BC'])
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

filename = 'sat50_reliability_'+ALGO+'_sampled'+str(SAMPLE_STEP)+'_rep'+str(NB_REPETITIONS)+'.csv'
print('\nExporting to', EXPORT_PATH+filename)
results_df.to_csv(EXPORT_PATH+filename, sep=',')