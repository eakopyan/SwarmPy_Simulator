import numpy as np
from math import dist 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


#==============================================================================================

class Node:
    """
    Node class, representing a satellite in the swarm. 
    """
    
    def __init__(self, id, x=0.0, y=0.0, z=0.0):
        """
        Node object constructor
        
        Args:
            id (int): the ID number of the satellite (mandatory)
            x (float, optional): the x-coordinate of the satellite. Defaults to 0.0.
            y (float, optional): the y-coordinate of the satellite. Defaults to 0.0.
            z (float, optional): the z-coordinate of the satellite. Defaults to 0.0.
        """
        self.id = int(id)
        self.x = float(x)
        self.y = float(y) 
        self.z = float(z) 
        self.neighbors = [] # List(Node), list of neighbor nodes to the node
        self.state = 0 # Message propagation state: 0 = no message received, 1 = message bearer, -1 = message transmitted
        
    def __str__(self):
        """
        Node object descriptor
        
        Returns:
            str: a string description of the node
        """
        nb_neigh = len(self.neighbors)
        return f"Node ID {self.id} ({self.x},{self.y},{self.z}) has {nb_neigh} neighbor(s)"
    
    def add_neighbor(self, node:Node):
        """
        Function to add a node to the neighbor list of the node unless it is already in its list.
        
        Args:
            node (int): the node to add
        """
        if node not in self.neighbors:
            self.neighbors.append(node)
        
    def remove_neighbor(self, node:Node):
        """
        Function to remove a node from the neighbor list of the node unless it is not in its list.
        
        Args:
            node (Node): the node to remove
        """
        if node in self.neighbors:
            self.neighbors.remove(node)
            
    def degree(self):
        """
        Function to compute the degree (aka the number of neighbors) of the node.
        
        Returns:
            int: the length of the neighbor list of the node
        """
        return len(self.neighbors)
    
    def compute_dist(self, node:Node):
        """
        Function to compute the Euclidean distance between two nodes.
        
        Args:
            node (Node): the node to compute the distance with

        Returns:
            float: the Euclidean distance between the two nodes
        """
        return dist((self.x, self.y, self.z) , (node.x, node.y, node.z))
    
    def is_neighbor(self, node:Node, connection_range=0):
        """
        Function to verify whether two nodes are neighbors or not, based on the connection range. 
        Either adds or removes the second node from the neighbor list of the first.
        
        Args:
            node (Node): the second node to analyse
            connection_range (int, optional): the maximum distance between two nodes to establish a connection. Defaults to 0.

        Returns:
            int: 1 if neighbors, 0 if not.
        """
        if node.id != self.id:
            if self.compute_dist(node) <= connection_range:
                self.add_neighbor(node)
                return 1 
            self.remove_neighbor(node)
        return 0
    
    def epidemic(self):
        """
        Function to simulate an epidemic propagation from the node to all its neighbors.
        The function affects the state attribute of the Node objects:
            0: the node has not received any message yet
            1: the node is currently bearing a message
            -1: the node has transmitted its message
        """
        nb_transmissions = 0
        for node in self.neighbors:
            if node.state == 0: # Previous state: the node has not received the message yet
                node.state = 1 # New state: the node receives the message now
                nb_transmissions += 1
        if nb_transmissions > 0: 
            self.state = -1 # the node has transmitted the message to at least one neighbor. Else, keeps it till next connection
        
    
    
#==============================================================================================

class Swarm:
    """
    Swarm object, representing a swarm of nanosatellites.
    """
    
    def __init__(self, connection_range=0, nodes=[]):
        """
        Swarm object constructor
        
        Args:
            connection_range (int, optional): the maximum distance between two nodes to establish a connection. Defaults to 0.
            nodes (list, optional): list of Node objects within the swarm. Defaults to [].
        """
        self.connection_range = connection_range
        self.nodes = nodes 
        
    def __str__(self):
        """
        Swarm object descriptor
        
        Returns:
            str: the string description of the swarm
        """
        nb_nodes = len(self.nodes)
        return f"Swarm of {nb_nodes} node(s), connection range: {self.connection_range}"
    
    def add_node(self, node:Node):
        """
        Function to add a node to the swarm unless it is already in.

        Args:
            node (Node): the node to add
        """
        if node not in self.nodes:
            self.nodes.append(node)
        
    def remove_node(self, node:Node):
        """
        Function to remove a node from the swarm unless it is already out.

        Args:
            node (Node): the node to remove
        """
        if node in self.nodes:
            self.nodes.remove(node)
        
    def degree(self):
        """
        Function to compute the degree (aka the number of neighbors) of each node within the swarm.

        Returns:
            list: list of node degrees (int)
        """
        return [node.degree() for node in self.nodes]
    
    def neighbor_matrix(self, connection_range=None):
        """
        Function to compute the neighbor matrix of the swarm.
        If two nodes are neighbors (according to the given connection range), the row[col] equals 1. Else 0.

        Args:
            connection_range (int, optional): the connection range of the swarm. Defaults to None.

        Returns:
            list: the 2-dimensional neighbor matrix formatted as matrix[row][column]
        """
        matrix = []
        if not connection_range:
            connection_range=self.connection_range # Use the attribute of the Swarm object if none specified
        for node in self.nodes:
            matrix.append([node.is_neighbor(nb,connection_range) for nb in self.nodes])
        return matrix
    
    def get_node_by_id(self, id:int):
        """
        Function to retrieve a Node object in the swarm from its ID.

        Args:
            id (int): the ID of the node

        Returns:
            Node: the Node object with the corresponding ID
        """
        for node in self.nodes:
            if node.id == id:
                return node
    
    def reset_state(self):
        """
        Function to reset all nodes states to 0 for the simulation of a message propagation.
        """
        for node in self.nodes:
            node.state = 0
            
    def get_swarm_state(self):
        """
        Function to retrieve the current state of each node of the swarm as a dict object.

        Returns:
            dict: (key, value) is (node ID, node state)
        """
        state = {}
        for node in self.nodes:
            state[node.id] = node.state
        return state
   
    def epidemic(self, ps = None):
        """
        Function to simulate an epidemic message propagation in the swarm.

        Args:
            ps (dict, optional): the previous (or initial) state of the swarm. Defaults to None.
        """
        for node_id, state in ps.items():
            node = self.get_node_by_id(node_id)
            node.state = state
            if state == 1: # Message bearer
                node.epidemic() 
            
        
    def plot(self, t:int):
        """
        Function to create a 3D-plot of the swarm at a given timestamp. 
        Visualizes the message propagation with 3 colors: 
            blue: the node has no message (state 0)
            red: the node carries the message (state 1)
            green: the node has transmitted its message (state -1)

        Args:
            t (int): timestamp of the simulation
        """
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
