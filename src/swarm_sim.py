import numpy as np
from math import dist 


#==============================================================================================

class Node:
    
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y) 
        self.z = float(z) 
        self.neighbors = []
        
    def __str__(self):
        nb_neigh = len(self.neighbors)
        return f"Node ({self.x},{self.y},{self.z}) has {nb_neigh} neighbor(s)"
    
    def add_neighbor(self, node):
        if node not in self.neighbors:
            self.neighbors.append(node)
        
    def remove_neighbor(self, node):
        if node in self.neighbors:
            self.neighbors.remove(node)
            
    def degree(self):
        return len(self.neighbors)
    
    def compute_dist(self, node):
        return dist((self.x, self.y, self.z) , (node.x, node.y, node.z))
    
    def is_neighbor(self, node, connection_range=0):
        if node != self:
            if self.compute_dist(node) <= connection_range:
                self.add_neighbor(node)
                return 1 
            self.remove_neighbor(node)
        return 0
    
    
    
#==============================================================================================

class Swarm:
    
    def __init__(self, connection_range=0, nodes=[]):
        self.connection_range = connection_range
        self.nodes = nodes
        
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
   
    
#==============================================================================================
