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
def rmse(data, ref=None):
    """
    This function calculates the Root Mean Square Error (RMSE) between the observed distribution and a reference value.

    Parameters:
    data (list or numpy array): A list or numpy array containing the observed data points.
    ref (float, optional): A reference value to compare the observed distribution with. Defaults to the mean of the observed data.

    Returns:
    float: The RMSE value, which represents the standard deviation of the differences between the observed data and the reference value.

    Example:
    >>> data = [1, 2, 3, 4, 5]
    >>> ref = 3
    >>> rmse(data, ref)
    0.8164965809277461
    """
    if ref is None:
        ref = np.mean(data)
    errors = [(e - ref) ** 2 for e in data]
    ratio = sum(errors) / len(data)
    return np.sqrt(ratio)


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
        
        
#============================= REFERENCE METRICS ================================
# Reference temporal evolution: average strength (AS), average clustering coefficient (ACC)
strength_ref, acc_ref = [], []

for t in np.arange(0, REVOLUTION, SAMPLE_STEP):
    strength_ref.append(np.mean(swarm_data[t].strength()))
    acc_ref.append(np.mean(list(swarm_data[t].cluster_coef(weight='cost').values())))

        
        
#============================== GRAPH DIVISION ==================================
rmse_dict = {
    'Timestamp':[],
    'Nb groups':[],
    'RMSE AS':[],
    'RMSE ACC':[]
}


ALGO = 'FFD'
print('\nDivision algorithm:', ALGO, '\t\tNumber of repetitions:', NB_REPETITIONS)

with tqdm(total=len(NB_GROUPS), desc='Number of groups') as group_bar:
    for nb_group in NB_GROUPS:
        for rep in range(NB_REPETITIONS):
            swarm_data[0].reset_groups()
            groups = swarm_data[0].FFD(n=nb_group, s=rep+1)# <==================== ALGO CHOICE 
            
            for t in np.arange(0, REVOLUTION, SAMPLE_STEP):
                swarm = swarm_data[t]
                group_strength = [] 
                group_acc = []

                for group_id, node_list in groups.items():
                    if len(node_list)>0:
                        group_strength.append(np.mean(swarm.strength(node_list)))
                        group_acc.append(np.mean(list(swarm.cluster_coef(node_list, weight='cost').values())))
                    
                rmse_strength = rmse(group_strength, strength_ref[int(t/SAMPLE_STEP)]) 
                rmse_acc = rmse(group_acc, acc_ref[int(t/SAMPLE_STEP)]) 
                
                rmse_dict['Timestamp'].append(t)
                rmse_dict['Nb groups'].append(nb_group)
                rmse_dict['RMSE AS'].append(rmse_strength)
                rmse_dict['RMSE ACC'].append(rmse_acc)      
        group_bar.update(1)
        
        
#===================================== EXPORT DATA ===================================        
df = pd.DataFrame(rmse_dict)
filename = 'sat50_RMSE_'+ALGO+'_sampled'+str(SAMPLE_STEP)+'_rep'+str(NB_REPETITIONS)+'.csv'
print('\nExporting to', EXPORT_PATH+filename)
df.to_csv(EXPORT_PATH+filename, sep=',')