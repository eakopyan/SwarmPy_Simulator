# coding: utf-8

#============================ IMPORTS ======================================
import numpy as np
import pandas as pd
import networkx as nx
import os
from tqdm import tqdm
from random import sample
from swarm_sim import *


#========================== GLOBAL VARIABLES ==============================
PATH_MALTE = 'C:\\Users\\EAkopyan\\Documents\\SwarmPy_Simulator'
PATH_COLIBRI = 'C:\\Users\\ankoc\\Documents\\Thèse\\SwarmPy_Simulator'

CONNECTION_RANGE = 30 # km
NB_NODES = 50
NB_REPETITIONS = 30
PATH = PATH_MALTE


#============================= FUNCTIONS ==================================
def add_data_row(data_dict,src,dst,nb_sp,spl,disp):
        data_dict['Source'].append(src)
        data_dict['Dest'].append(dst)
        data_dict['Number of SP'].append(nb_sp)
        data_dict['SP length'].append(spl)
        data_dict['Disparity'].append(disp)
        

def FFD(swarm, n=10, p=0.7, s=1):
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
        return groups
    

def group_betweeness_centrality(groups, graph):
    dict = group_shortest_paths(groups, graph)
    shortest_paths = dict['Shortest paths']
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
        bc = bc/(origin_destination_pairs(groups)) #Normalize over all possible pairs
        bc_dict['Node'].append(node)
        bc_dict['BC'].append(bc)
    return bc_dict
    

def group_shortest_paths(groups, graph):
    visited_pairs = []
    data_dict = {
        'Group':[],
        'Source':[],
        'Dest':[],
        'Shortest paths':[]
    }
    for group_id, group_nodes in groups.items():
        for ni in group_nodes:
            src_id = ni.id
            for nj in group_nodes:
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


def kronecker_delta(ni, nj):
    if ni.group==nj.group:
        return 1
    return 0


def MDRW(swarm, n=10, s=1):
    sources = sample(swarm.nodes, n) # Initial random sources
    groups = {} # Dict(group ID:list(nodes))
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
    return groups


def modularity(swarm, neighbor_matrix, nb_edges):
    element = 0
    for ni in swarm.nodes:
        for nj in swarm.nodes:
            element += (neighbor_matrix[ni.id][nj.id] - ni.degree()*nj.degree()/(2*nb_edges)) * kronecker_delta(ni, nj)
    modularity = element / (2*nb_edges)
    return modularity


def origin_destination_pairs(groups):
    nb_pairs = 0
    for group in groups.values():
        group_size = len(group)
        nb_pairs += group_size*(group_size-1)/2
    return nb_pairs


def pair_disparity(shortest_paths:list, spl:int):
    if len(shortest_paths)==1:
        return 0.0
    path_graphs = {}
    for i,path in enumerate(shortest_paths):
        path_graphs[i] = nx.path_graph(path)
    disparity = 0
    pairs = []
    for idx1,p1 in path_graphs.items():
        for idx2,p2 in path_graphs.items():
            if idx1 != idx2 and set([idx1,idx2]) not in pairs:
                pairs.append(set([idx1,idx2]))
                its = nx.intersection(p1,p2)
                disp = 1 - nx.number_of_edges(its)/spl
                disparity += disp
    return disparity/len(pairs)


def RND(swarm, clist=range(10), s=1):
    groups = {} # Dict(group ID:list(nodes))
    for i, node in enumerate(swarm.nodes):
        node.random_group(clist, s*i)
    for group_id in clist:
        groups[group_id] = [node for node in swarm.nodes if node.group==group_id]
    return groups



#============================ INITIALIZE TOPOLOGY ==============================
print('Importing topology...')
topo_path = 'data\\Topologies'
file = 'topology_connected_sat50.csv'
df_low = pd.read_csv(os.path.join(PATH, topo_path, file), sep=',', header=0, index_col='sat_id')

swarm = Swarm(
    connection_range=CONNECTION_RANGE,
    nodes=[Node(i, df_low.iloc[i].x, df_low.iloc[i].y, df_low.iloc[i].z) for i in list(df_low.index.values)]
)
print(swarm)
print('Computing neighbor matrix.')
neighbor_matrix = swarm.neighbor_matrix()

#==================================== BEGIN =====================================
print('\nConverting to NetworkX graph:')
graph = swarm.swarm_to_nxgraph()
print(graph)

# Dict to store data (convert later into pd.DataFrame)
data_dict = {
    'Source':[],
    'Dest':[],
    'Number of SP':[],
    'SP length':[],
    'Disparity':[]
}

# ========================== CHOICE OF ALGORITHM HERE =============================
algo = 'FFD'
print('\nPerforming graph division:', algo, '\t\tNumber of repetitions:', NB_REPETITIONS)
print('Initializing redundancy analysis...')

connectivities = []
modularities = []
nb_critical_nodes = []
routing_costs = []
network_efficiencies = []

with tqdm(total=NB_REPETITIONS, desc='Random iterations') as pbar:
    for rep in range(NB_REPETITIONS):
        swarm.reset_groups()
        # =================== CHANGE ALGORITHM HERE ==========================
        groups = FFD(swarm, s=rep)

        visited_pairs = []
        total_spl = 0
        pair_efficiency = 0.0
        
        for group_id, group_nodes in groups.items():
            for ni in group_nodes:
                src_id = ni.id
                for nj in group_nodes:
                    dst_id = nj.id
                    if dst_id != src_id and set((src_id,dst_id)) not in visited_pairs:
                        pair_efficiency += nx.efficiency(graph, src_id, dst_id)
                        if nx.has_path(graph, src_id, dst_id):
                            visited_pairs.append(set((src_id,dst_id)))
                            spl = nx.shortest_path_length(graph, source=src_id, target=dst_id)
                            shortest_paths = nx.all_shortest_paths(graph, src_id, dst_id)
                            list_paths = list(shortest_paths)
                            disparity = pair_disparity(list_paths, spl)
                            total_spl += spl
                    
                            add_data_row(data_dict, 
                                        src=src_id, 
                                        dst=dst_id, 
                                        nb_sp=len(list_paths),
                                        spl=spl, 
                                        disp=disparity)
                            
        nb_max = origin_destination_pairs(groups)
        
        connectivities.append(len(visited_pairs)/nb_max)
        modularities.append(modularity(swarm, neighbor_matrix, nx.number_of_edges(graph)))
        df_group = pd.DataFrame(group_betweeness_centrality(groups, graph))
        nb_critical_nodes.append(len(df_group[df_group['BC']>=0.05]['BC']))
        
        routing_costs.append(total_spl*2)
        network_efficiencies.append(pair_efficiency/nb_max)
        
        pbar.update(1)


df = pd.DataFrame(data_dict)

print('\n\n=============================================================')

print('\nResults on resilience:')
print('\tConnectivity: ' + str(round(np.mean(connectivities)*100, 1))+'%')
print('\tAverage number of SP: ' + str(round(df['Number of SP'].mean(), 1))+' paths')
print('\tAverage disparity: ' + str(round(df['Disparity'].mean()*100, 1))+'%')
print('\tAverage modularity: ' + str(round(np.mean(modularities)*100, 1))+'%')
print('\tAverage number of critical nodes: ' + str(round(np.mean(nb_critical_nodes), 1))+' node(s)')

print('\nResults on robustness:')
print('\tAverage routing cost: ' + str(int(np.mean(routing_costs)))+' transmissions')
print('\tAverage network efficiency: ' + str(round(np.mean(network_efficiencies)*100, 1))+'%')


export_path = 'output\\data'
filename = 'path_redundancy_'+algo+'_connected_sat50_rep'+str(NB_REPETITIONS)+'.csv'
df.to_csv(os.path.join(PATH, export_path, filename), sep=',')
print('\nExporting to', os.path.join(PATH, export_path, filename))