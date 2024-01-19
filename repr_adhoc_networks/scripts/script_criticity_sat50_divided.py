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

NB_REPETITIONS = 30
CONNECTION_RANGE = 30 # km
NB_NODES = 50
DURATION = 8641 # Number of data rows, not time!
REVOLUTION = 1800 # Number of data rows
SAMPLE_STEP = 12 # Take one out of 12 samples (alleviates calculations)


#============================= FUNCTIONS ==================================
def FFD(swarm, n=10, p=0.7, s=1, by_id=False):
    sources = sample(swarm.nodes, n) # Initial random sources
    groups = {} # Dict(group ID:list(nodes))
    for group_id,src in enumerate(sources): # Initialize swarms
        src.set_group(group_id)
        groups[group_id] = [src]
    free_nodes = [n for n in swarm.nodes if n.group==-1]
    burning_nodes = sources
    next_nodes = []
    while free_nodes: # Spread paths from each burning node in parallel
        for bn in burning_nodes:
            if not free_nodes:
                break
            free_neighbors = set(free_nodes).intersection(bn.neighbors)
            if free_neighbors: # At least one unassigned neighbor
                nodes = bn.proba_walk(p, s) # Next node(s)
            else:
                nodes = [swarm.random_jump(s)] # If no neighbor, perform random jump in the graph
            for n in nodes:
                n.set_group(bn.group)
                groups[bn.group].append(n)
                free_nodes.remove(n)
                next_nodes.append(n)
        burning_nodes = next_nodes
    if by_id:
        groups_by_id = {}
        for gid, nodes in groups.items():
            groups_by_id[gid] = [n.id for n in nodes]
        return groups_by_id
    return groups


def group_betweeness_centrality(groups, graph, tsp, by_id=False):
    dict = group_shortest_paths(groups, graph, by_id)
    shortest_paths = dict['Shortest paths']
    for paths in shortest_paths:
        for path in paths:
            del path[0] # Discard endpoints
            del path[-1]
    bc_dict = {
        'Node':[],
        'NBC':[]
    }
    for node in graph.nodes:
        bc = 0
        for paths in shortest_paths:
            sp_node = [path for path in paths if node in path]
            bc += len(sp_node)/len(paths)
        bc = bc/(origin_destination_pairs(groups)) #Normalize over all possible pairs
        bc_dict['Node'].append(node)
        bc_dict['NBC'].append(bc)
    df_bc = pd.DataFrame(bc_dict)
    df_bc['Timestamp'] = tsp
    return df_bc
    

def group_shortest_paths(groups, graph, by_id=False):
    visited_pairs = []
    data_dict = {
        'Group':[],
        'Source':[],
        'Dest':[],
        'Shortest paths':[]
    }
    for group_id, group_nodes in groups.items():
        for ni in group_nodes:
            if by_id:
                src_id = ni
            else:
                src_id = ni.id
            for nj in group_nodes:
                if by_id:
                    dst_id = nj
                else:
                    dst_id = nj.id
                if set((src_id,dst_id)) not in visited_pairs:
                    visited_pairs.append(set((src_id,dst_id)))
                    if dst_id != src_id and nx.has_path(graph, src_id, dst_id):
                        # Compute shortest paths
                        gen_paths = nx.all_shortest_paths(graph, source=src_id, target=dst_id)
                        shortest_paths = list(gen_paths)
                        data_dict['Group'].append(group_id)
                        data_dict['Source'].append(src_id)
                        data_dict['Dest'].append(dst_id)
                        data_dict['Shortest paths'].append(shortest_paths)
    return data_dict


def MIRW(swarm, n=10, s=1, by_id=False):
    sources = sample(swarm.nodes, n) # Initial random sources
    groups = {} # Dict(group ID:list(Node))
    for group_id, src in enumerate(sources): # Initialize swarms
        src.set_group(group_id)
        groups[group_id] = [src]
    free_nodes = [n for n in swarm.nodes if n.group==-1]
    while free_nodes: # Spread paths
        for group_id in groups.keys():
            n_i = groups[group_id][-1] # Current node
            free_neighbors = set(free_nodes).intersection(n_i.neighbors)
            if free_neighbors: # At least one unassigned neighbor
                n_j = n_i.random_walk(s*group_id) # Next node
            else:
                n_j = swarm.random_jump(s) # If no neighbor, perform random jump in the graph
            n_j.set_group(n_i.group)
            groups[group_id].append(n_j)
            free_nodes.remove(n_j)
    if by_id:
        groups_by_id = {}
        for gid, nodes in groups.items():
            groups_by_id[gid] = [n.id for n in nodes]
        return groups_by_id
    return groups


def origin_destination_pairs(groups):
    nb_pairs = 0
    for group in groups.values():
        group_size = len(group)
        nb_pairs += group_size*(group_size-1)/2
    return nb_pairs


def RND(swarm, clist=range(10), s=1, by_id=False):
    groups = {} # Dict(group ID:list(nodes))
    for i, node in enumerate(swarm.nodes):
        node.random_group(clist, s*i)
    for group_id in clist:
        groups[group_id] = [node for node in swarm.nodes if node.group==group_id]
    if by_id:
        groups_by_id = {}
        for gid, nodes in groups.items():
            groups_by_id[gid] = [n.id for n in nodes]
        return groups_by_id
    return groups



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

algo = 'FFD' # <==================== ALGO CHOICE 
print('\nPerforming graph division:', algo, '\t\tNumber of repetitions:', NB_REPETITIONS)

for rep in range(NB_REPETITIONS):
    swarm_data[0].reset_groups()
    groups = FFD(swarm_data[0], s=rep, by_id=True) # <==================== ALGO CHOICE 
    nb_max = origin_destination_pairs(groups)
    group_assignment = {}
    for node in swarm_data[0].nodes:
        group_assignment[node.id] = node.group

    with tqdm(total=REVOLUTION/SAMPLE_STEP, desc='NBC distribution '+str(rep)) as pbar:
        for t in np.arange(0, REVOLUTION, SAMPLE_STEP):
            graph = topo_graphs[t]
            df_temp = group_betweeness_centrality(groups, graph, t, by_id=True)
            final_df = pd.concat([final_df, df_temp], ignore_index=True)

            pbar.update(1)
 

#=============================== EXPORTING RESULTS ============================

print(final_df.head())
print(final_df.shape[0], 'rows')

filename = 'sat50_criticity_'+algo+'_sampled'+str(SAMPLE_STEP)+'_rep'+str(NB_REPETITIONS)+'.csv'
print('\nExporting to', os.path.join(EXPORT_PATH, filename))
final_df.to_csv(os.path.join(EXPORT_PATH, filename), sep=',')