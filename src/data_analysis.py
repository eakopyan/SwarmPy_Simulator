import pandas as pd
import numpy as np
from math import dist 
from tqdm import tqdm

#Calcul distance min et max pour chaque satellite
def compute_dist(sat_a:pd.DataFrame, sat_b:pd.DataFrame, t=0):
    pa = (sat_a[t].x, sat_a[t].y, sat_a[t].z)
    pb = (sat_b[t].x, sat_b[t].y, sat_b[t].z)
    return dist(pa,pb)

#Create lists of distances for each sat wrt swarm at timestamp t
def compute_swarm_distances(satellites:dict, t=0):
     sat_dist_instant = {}
     for k in satellites.keys():
          sat_dist_instant[k] = [compute_dist(satellites[k], satellites[i], t) for i in range(0,100) if i!=k]
     return sat_dist_instant
    
#Create lists of distances for each sat wrt swarm over a simulation period
#Return format: dict of timestamps, containing dict of satellites, containing list of distances
def compute_swarm_mobility(satellites:dict, timeslot=1):
     mobility = {}
     with tqdm(total=timeslot, desc='Computing distances') as pbar:
          for t in range(0,timeslot):
               mobility[t] = compute_swarm_distances(satellites, t)
               pbar.update(1)
     return mobility

def analyse_distances(data:dict):
     mins = [np.min(df[k]) for df in data.values() for k in df.keys()] #Min distance for each sat
     maxes = [np.max(df[k]) for df in data.values() for k in df.keys()] #Max distance for each sat
     means = [np.mean(df[k]) for df in data.values() for k in df.keys()] #Mean distance for each sat
     print('Minimum distance in swarm:', np.mean(mins))
     print('Lower bound:', np.min(mins))
     print('Upper bound:', np.max(mins))
     print('\nAverage distance in swarm:', np.mean(means))
     print('Lower bound:', np.min(means))
     print('Upper bound:', np.max(means))
     print('\nMaximum distance in swarm:', np.mean(maxes))
     print('Lower bound:', np.min(maxes))
     print('Upper bound:', np.max(maxes))
     
     
#============ NEIGHBORS DISCOVERY =============

def is_neighbor(sat_a:pd.DataFrame, sat_b:pd.DataFrame, scope=0, t=0):
     if compute_dist(sat_a, sat_b, t) <= scope:
          return 1 
     return 0

#Create instantaneous list of neighbors according to given range
def find_neighbors(satellites:dict, scope=0, t=0):
     neighbors = {}
     for k in satellites.keys():  
          neighbors[k] = [is_neighbor(satellites[k], satellites[i], scope, t) for i in satellites.keys()]
     return neighbors

#Create lists of neighbors for each sat wrt swarm over a simulation period
#Return format: dict of timestamps, containing dict of satellites, containing list of neighbors
def find_swarm_neighbors(satellites:dict, scope=0, timeslot=1):
     neighbors = {}
     with tqdm(total=timeslot, desc='Finding neighbors') as pbar:
          for t in range(0,timeslot):
               neighbors[t] = find_neighbors(satellites, scope, t)
               pbar.update(1)
     return neighbors

def compute_disponibility(neighbors:dict, nb_nodes=100, timeslot=1):
     node_disp = {}
     for k in range(nb_nodes):
          ct = {}
          for i in range(nb_nodes):
               ct[i] = sum(neighbors[t][k][i] for t in range(timeslot)) / float(timeslot)*100
          node_disp[k] = ct
     return node_disp