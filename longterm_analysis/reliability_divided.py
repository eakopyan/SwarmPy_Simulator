# coding: utf-8

#============================ IMPORTS ======================================
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
from swarm_sim import *



#========================== GLOBAL VARIABLES ==============================
PATH = '..\\data\\v10-swarm-50s-pert\\coords_LLO-'
EXPORT_PATH = 'longterm_analysis\\output\\data\\'

NB_NODES = 50       # Nombre de satellites (noeuds)
DURATION = 3601    # Nombre total d'échantillons disponibles
REVOLUTION = 1800   # Nombre d'échantillons pour 1 révolution en orbite lunaire
MONTHS = [1,12] # np.arange(1,13)
SAMPLE_FREQ = 0.1   # Fréquence d'échantillonnage (Hz) : toutes les 10 secondes
CONNECTION_RANGE = 30    # Portée de connexion (km)
PROPAGATION = 'J4'
ROW_DATA_START = 7

SAMPLE_STEP = 30    # Fréquence de ré-échantillonnage pour ne pas analyser toutes les topologies
NB_REPETITIONS = 30 # Nombre de répétitions aléatoires et indépendantes des algorithmes de division
NB_GROUPS = 10      # Division en 10 groupes


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
    """if len(shortest_paths)==1:
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
    return disparity/len(pairs)"""
    return 0.0


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
swarm_months = {}
ALGO = 'FFD' # RND, MIRW, FFD or KMeans
print('\nDivision algorithm:', ALGO, '\t\tNumber of repetitions:', NB_REPETITIONS)
        
for m in MONTHS:
    print('\nExtracting month', m)
    satellites = {} # Dict(sat_id: DataFrame)
    with tqdm(total=NB_NODES, desc='Extracting data') as pbar:
        for i in range(NB_NODES):
            df = pd.read_csv(PATH+str(i)+'-'+PROPAGATION+'_'+str(m)+'.csv', skiprows= lambda x: x<ROW_DATA_START, header=0)
            satellites[i] = df
            pbar.update(1)
            
    swarm_data = {} # Dict{timestamp: Swarm}
    with tqdm(total = REVOLUTION, desc = 'Converting to Swarm') as pbar:
        for t in range(REVOLUTION):
            swarm_data[t] = Swarm(
                connection_range = CONNECTION_RANGE, 
                nodes = [Node(id, sat['xF[km]'].iloc[t], sat['yF[km]'].iloc[t], sat['zF[km]'].iloc[t]) for id,sat in satellites.items()]
                )
            pbar.update(1)
            
    # Ré-échantillonner les topologies (réduit les calculs)
    swarm_topo = {}
    for t in np.arange(0, REVOLUTION, SAMPLE_STEP):
        swarm_topo[t] = swarm_data[t]
        
    neighbor_matrices = {} # Dict{timestamp: matrix}
    with tqdm(total=REVOLUTION/SAMPLE_STEP, desc='Computing neighbor matrices') as pbar:
        for t in swarm_topo.keys():
            neighbor_matrices[t] = swarm_topo[t].neighbor_matrix(weighted=True)
            pbar.update(1)
            
    # Création des graphes associés  
    with tqdm(total=REVOLUTION/SAMPLE_STEP, desc='Generating graphs') as pbar:
        for t in swarm_topo.keys():
            swarm_topo[t].create_graph()
            pbar.update(1)
            
    # Enlever les ISL trop chers de l'essaim (ceux dont le coût est supérieur au coût du plus court chemin)
    with tqdm(total=REVOLUTION/SAMPLE_STEP, desc = 'Removing expensive edges') as pbar:
        for t in swarm_topo.keys():
            swarm_topo[t].remove_expensive_edges()
            pbar.update(1)
    
    swarm_months[m] = swarm_topo
            
swarm_months[1][0].reset_groups()
if ALGO=='RND':
    groups = swarm_months[1][0].RND(n=NB_GROUPS, s=1)
elif ALGO=='MIRW':
    groups = swarm_months[1][0].MIRW(n=NB_GROUPS, s=1) # <==================== ALGO CHOICE 
elif ALGO=='FFD':
    groups = swarm_months[1][0].FFD(n=NB_GROUPS, s=1)
elif ALGO=='KMeans':
    kmeans = KMeans(n_clusters=NB_GROUPS).fit([[n.x, n.y, n.z] for n in swarm_months[1][0].nodes])
    groups = {}
    for i in range(NB_GROUPS):
        groups[i] = [node.id for node in swarm_months[1][0].nodes if kmeans.labels_[node.id]==i]
 
       
for m in MONTHS:
    # Dict to store data (convert later into pd.DataFrame) (reinit each month)
    monthly_data = {
        'Timestamp':[],
        'RFlow':[],
        'RCost':[],
        'Efficiency':[],
        'Redundancy':[],
        'Disparity':[],
        'Criticity': []
    }
    print('\nProcessing month', m)
    swarm_data = swarm_months[m]
    nb_flow = swarm_flow_nb(groups)

    with tqdm(total=REVOLUTION/SAMPLE_STEP, desc='Temporal evolution') as pbar:
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

            add_data_row(monthly_data,
                        t,
                        rflow,
                        rcost, 
                        swarm_efficiency,  
                        np.mean(redundancies),
                        np.mean(disparities),
                        crit)
            pbar.update(1)
        
    #===================================== EXPORT DATA ===================================        
    results_df = pd.DataFrame(monthly_data)
    print(results_df.head())
    print(results_df.shape[0], 'rows')

    filename = 'sat50_reliability_'+ALGO+'_year'+str(m)+'.csv'
    results_df.to_csv(EXPORT_PATH+filename, sep=',')
    print('\nExported to', EXPORT_PATH+filename)