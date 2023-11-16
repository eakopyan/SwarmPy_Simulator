# coding: utf-8

#============================ IMPORTS ======================================
import numpy as np
import pandas as pd
import networkx as nx
import os
from swarm_sim import *


#========================== GLOBAL VARIABLES ==============================
PATH = 'C:\\Users\\EAkopyan\\Documents\\SwarmPy_Simulator-3'

CONNECTION_RANGE = 30 # km
NB_NODES = 50


#============================= FUNCTIONS ==================================
def add_data_row(data_dict,src,dst,nb_sp,spl,disp):
        data_dict['Source'].append(src)
        data_dict['Dest'].append(dst)
        data_dict['Number of SP'].append(nb_sp)
        data_dict['SP length'].append(spl)
        data_dict['Disparity'].append(disp)
        

def swarm_betweeness_centrality(graph):
    bc = nx.betweenness_centrality(graph)
    bc_dict = {
        'Node':list(bc.keys()),
        'BC':list(bc.values())
    }
    return bc_dict


def kronecker_delta(ni, nj):
    if ni.group==nj.group:
        return 1
    return 0


def modularity(swarm, neighbor_matrix, nb_edges):
    element = 0
    for ni in swarm.nodes:
        for nj in swarm.nodes:
            element += (neighbor_matrix[ni.id][nj.id] - ni.degree()*nj.degree()/(2*nb_edges)) * kronecker_delta(ni, nj)
    modularity = element / (2*nb_edges)
    return modularity


def origin_destination_pairs():
    return NB_NODES*(NB_NODES-1)/2


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
print('\nNo graph division here.')
print('Initializing redundancy analysis on original graph...')

connectivities = []
modularities = []
nb_critical_nodes = []
routing_costs = []
network_efficiencies = []


visited_pairs = []
total_spl = 0
for ni in swarm.nodes:
    src_id = ni.id
    for nj in swarm.nodes:
        dst_id = nj.id
        if dst_id != src_id and set((src_id,dst_id)) not in visited_pairs and nx.has_path(graph, src_id, dst_id):
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
nb_max = origin_destination_pairs()

connectivities.append(len(visited_pairs)/nb_max)
modularities.append(modularity(swarm, neighbor_matrix, nx.number_of_edges(graph)))
df_bc = pd.DataFrame(swarm_betweeness_centrality(graph))
nb_critical_nodes.append(len(df_bc[df_bc['BC']>=0.05]['BC']))

routing_costs.append(total_spl*2)
network_efficiencies.append((1/total_spl) * nb_max)


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
filename = 'path_redundancy_undivided_connected_sat50.csv'
df.to_csv(os.path.join(PATH, export_path, filename), sep=',')
print('\nExporting to', os.path.join(PATH, export_path, filename))