import numpy as np
from math import dist 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


#==============================================================================================

class Node:
    
    def __init__(self, id, x=0.0, y=0.0, z=0.0):
        self.id = int(id)
        self.x = float(x)
        self.y = float(y) 
        self.z = float(z) 
        self.neighbors = [] # List of nodes
        self.state = 0 # Message propagation
        
    def __str__(self):
        nb_neigh = len(self.neighbors)
        return f"Node ID {self.id} ({self.x},{self.y},{self.z}) has {nb_neigh} neighbor(s)"
    
    def add_neighbor(self, node):
        if node not in self.neighbors:
            self.neighbors.append(node)
        
    def remove_neighbor(self, node):
        if node in self.neighbors:
            self.neighbors.remove(node.id)
            
    def degree(self):
        return len(self.neighbors)
    
    def compute_dist(self, node):
        return dist((self.x, self.y, self.z) , (node.x, node.y, node.z))
    
    def is_neighbor(self, node, connection_range=0):
        if node.id != self.id:
            if self.compute_dist(node) <= connection_range:
                self.add_neighbor(node)
                return 1 
            self.remove_neighbor(node)
        return 0
    
    def epidemic(self): # PS = Previous State, NS = New State
        next_hop_count = 0 # Potential receivers
        for node in self.neighbors:
            if node.state == 0: # PS has not had the message yet
                node.state = 1 # NS has the message now
                next_hop_count += 1
        if next_hop_count > 0:
            self.state = -1 # NS has passed the message. If no neighbors, keeps it till next connection
        
    
    
#==============================================================================================

class Swarm:
    
    def __init__(self, connection_range=0, nodes=[]):
        self.connection_range = connection_range
        self.nodes = nodes # List of nodes
        
    def __str__(self):
        nb_nodes = len(self.nodes)
        return f"Swarm of {nb_nodes} node(s), connection range: {self.connection_range}"
    
    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)
        
    def remove_node(self, node):
        if node in self.nodes:
            self.nodes.remove(node)
        
    def degree(self):
        return [node.degree() for node in self.nodes]
    
    def neighbor_matrix(self, connection_range=None):
        matrix = []
        if not connection_range:
            connection_range=self.connection_range
        for node in self.nodes:
            matrix.append([node.is_neighbor(nb,connection_range) for nb in self.nodes])
        return matrix
    
    def get_node_by_id(self, id):
        for node in self.nodes:
            if node.id == id:
                return node
    
    def reset_state(self):
        for node in self.nodes:
            node.state = 0
            
    def get_swarm_state(self):
        state = {}
        for node in self.nodes:
            state[node.id] = node.state
        return state
   
    def epidemic(self, ps = None):
        for node_id, state in ps.items():
            node = self.get_node_by_id(node_id)
            node.state = state
            if state == 1: # Message bearer
                node.epidemic() 
            
        
    def plot(self, t):
        fig = plt.figure(figsize=(8,8))
        ax = plt.axes(projection='3d')

        x_data = [node.x for node in self.nodes]
        y_data = [node.y for node in self.nodes]
        z_data = [node.z for node in self.nodes]
        node_states = np.array([node.state for node in self.nodes])
        colormap = np.array(['blue','red','green'])
        ax.scatter(x_data, y_data, z_data, c=colormap[node_states])
        ax.set_title('Propagation at time '+str(t))
        
#==============================================================================================
