from typing import List
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from numpy.random import binomial
from math import dist 
from mpl_toolkits import mplot3d
from random import seed, randint, choice, sample


#==============================================================================================

class Node:
    """
    Node class, representing a satellite in the swarm. 
    """
    
    def __init__(self, id, x=0.0, y=0.0, z=0.0):
        """
        Node object constructor.
        
        Args:
            id (int): the ID number of the satellite (mandatory).
            x (float, optional): the x-coordinate of the satellite. Defaults to 0.0.
            y (float, optional): the y-coordinate of the satellite. Defaults to 0.0.
            z (float, optional): the z-coordinate of the satellite. Defaults to 0.0.
        
        Returns:
            Node: an instance of the Node class.
        """
        self.id = int(id)
        self.x = float(x)
        self.y = float(y) 
        self.z = float(z) 
        self.neighbors = {} # Dict(Node ID, weight), dict of neighbors of the node with corresponding ISL weight (defaults to 1)
        self.group = -1 # int, group ID to which belongs the node, defaults to -1 (unassigned)
        
    def __str__(self):
        """
        Node object descriptor.
        
        Returns:
            str: a string description of the node.
        """
        nb_neigh = len(self.neighbors.keys())
        return f"Node ID {self.id} ({self.x},{self.y},{self.z}) has {nb_neigh} neighbor(s)\tGroup: {self.group}"
    
    #*************** Common operations ****************
    def add_neighbor(self, node_id, cost=1):
        """ 
        Function to add a node from the swarm as a neighbor. 

        Args:
            node_id (int): the ID of the node to add. 
            cost (float, optional): the weight of the ISL between the two nodes. Defaults to 1.
        """
        self.neighbors[node_id] = cost
        
    def compute_dist(self, node):
        """
        Function to compute the Euclidean distance between two nodes.
        
        Args:
            node (Node): the node to compute the distance with.

        Returns:
            float: the Euclidean distance between the two nodes.
        """
        return dist((self.x, self.y, self.z) , (node.x, node.y, node.z))
    
    def is_neighbor(self, node, connection_range=0, weighted=False):
        """
        Function to verify whether two nodes are neighbors or not, based on the connection range. 
        Either adds or removes the second node from the neighbor list of the first.
        
        Args:
            node (Node): the second node to analyse.
            connection_range (int, optional): the maximum distance between two nodes to establish a connection. Defaults to 0.

        Returns:
            int: weight of the ISL, 0 if no ISL or same node.
        """
        cost = 0
        if node.id != self.id:
            dist = self.compute_dist(node)
            if weighted and dist <= 2*connection_range:
                cost = 4 # emission cost is proportionnal to the square of the distance: if the distance is doubled, the cost is multiplied by 4
                self.add_neighbor(node.id, cost)
            if dist <= connection_range:
                cost = 1
                self.add_neighbor(node.id, cost)
            if dist > 2*connection_range:
                self.remove_neighbor(node.id)
        return cost
    
    def remove_neighbor(self, node_id):
        """
        Function to remove a node from the neighbor list of the node unless it is not in its list.
        
        Args:
            node_id (int): the node to remove by ID.
        """
        if node_id in self.neighbors.keys():
            del self.neighbors[node_id]
     
    def set_group(self, c):
        """
        Function to appoint a group ID to the node.

        Args:
            c (int): group ID.
        """
        self.group = c
    
     
    #*********** Metrics ***************   
    def cluster_coef(self):
        """
        Function to compute the clustering coefficient of a node, which is defined as
        the existing number of edges between the neighbors of a node divided by the maximum
        possible number of such edges.

        Returns:
            float: the clustering coefficient of the node between 0 and 1.
        """
        dv = self.degree()
        max_edges = dv*(dv-1)/2
        if max_edges == 0:
            return 0
        edges = 0
        for v in self.neighbors:
            common_elem = set(v.neighbors).intersection(self.neighbors)
            edges += len(common_elem)
        return edges/(2*max_edges) # Divide by 2 because each edge is counted twice
                    
    def degree(self):
        """
        Function to compute the degree (aka the number of neighbors) of the node. The neighbor lists must be established before running
        this function.
        
        Returns:
            int: the length of the neighbor list of the node.
        """
        return len(self.neighbors)
                
    def k_vicinity(self, depth=1):
        """
        Function to compute the k-vicinity (aka the extended neighborhood) of the node.
        The k-vicinity corresponds to the number of direct and undirect neighbors within at most k hops from the node.

        Args:
            depth (int, optional): the number of hops for extension. Defaults to 1.

        Returns:
            int: the length of the extended neighbor list of the node.
        """
        kv = self.neighbors.copy()
        for i in range(depth-1):
            nodes = kv
            kv.extend([n for node in nodes for n in node.neighbors])
        return len(set(kv))   
    
    
    #*************** Sampling algorithms ****************
    def proba_walk(self, p:float, s=1, overlap=False):
        """
        Function to perform a probabilistic hop from the node to its neighbor(s), usually used with the Forest Fire algorithm (see Swarm object).
        Each neighbor node has a probability p to be chosen for the next hop.

        Args:
            p (float): the success probability between 0 and 1.
            s (int, optional): the random seed. Defaults to 1.
            overlap (bool, optional): if True, node groups are allowed to overlap. Defaults to False.

        Returns:
            list(Node): the list of neighbor nodes selected as next hops.
        """
        seed(s)
        search_list = self.neighbors
        if not overlap: # Restrain the search list to unassigned nodes
            search_list = [n for n in self.neighbors if n.group==-1]
        trial = binomial(1, p, len(search_list))
        nodes = [n for i,n in enumerate(search_list) if trial[i]==1] # Select the nodes that obtained a success above
        return nodes
    
    def random_group(self, clist, s=1):
        """
        Function to appoint a group ID chosen randomly from the input list.

        Args:
            clist (list(int)): the list of group IDs.
            s (int, optional): the random seed. Defaults to 1.
        """
        seed(s)
        self.set_group(choice(clist))
        
    def random_walk(self, s=1, overlap=False):
        """
        Function to perform a random walk from the current node. One of its neighbor nodes is chosen randomly as the next hop.

        Args:
            s (int, optional): the random seed for the experiment. Defaults to 1.
            overlap (bool, optional): if True, node groups are allowed to overlap. Defaults to False.

        Returns:
            Node: the neighbor node selected as the next hop.
        """
        seed(s)
        search_list = self.neighbors
        if not overlap: # Restrain the search list to unassigned nodes
            search_list = [n for n in self.neighbors if n.group==-1]
        return choice(search_list)
    
    
#==============================================================================================

class Swarm:
    """
    Swarm object, representing a swarm of nanosatellites.
    """
    
    def __init__(self, connection_range=0, nodes=[]):
        """
        Swarm object constructor.
        
        Args:
            connection_range (int, optional): the maximum distance between two nodes to establish a connection. Defaults to 0.
            nodes (list, optional): list of Node objects within the swarm. Defaults to [].
        """
        self.connection_range = connection_range
        self.nodes = nodes 
        self.graph = None
        
    def __str__(self):
        """
        Swarm object descriptor
        
        Returns:
            str: the string description of the swarm
        """
        nb_nodes = len(self.nodes)
        return f"Swarm of {nb_nodes} node(s), connection range: {self.connection_range}"
    
    #*************** Common operations ***************
    def add_node(self, node:Node):
        """
        Function to add a node to the swarm unless it is already in.

        Args:
            node (Node): the node to add.
        """
        if node not in self.nodes:
            self.nodes.append(node)
            if self.graph != None:
                self.graph.add_node(node.id, group=node.group)
            
    def distance_matrix(self):
        """
        Function to compute the Euclidean distance matrix of the swarm.

        Returns:
            list(list(float)): the 2-dimensional distance matrix formatted as matrix[node1][node2] = distance.
        """
        matrix = []
        for n1 in self.nodes:
            matrix.append([n1.compute_dist(n2) for n2 in self.nodes])
        return matrix
    
    
    def find_closest_neighbors(self, node, connection_range):
        coef = 0
        neighbors = []
        while len(neighbors)==0:
            coef += 1
            distances = [(n2, node.compute_dist(n2)) for n2 in self.nodes]
            neighbors = [d[0] for d in distances if d[1]<=coef*connection_range and d[1]>0]
        return (neighbors, coef)

    
    def get_node_by_id(self, id:int):
        """
        Function to retrieve a Node object in the swarm from its node ID.

        Args:
            id (int): the ID of the node.

        Returns:
            Node: the Node object with the corresponding ID.
        """
        for node in self.nodes:
            if node.id == id:
                return node
            
            
    def isolated_nodes(self):
        """
        Function to retrieve a list of nodes that have no neighbors within the given connection range.

        Args:
            self (Swarm): an instance of the Swarm class.

        Returns:
            list(Node): a list of nodes with no neighbors.
        """
        return [node for node in self.nodes if len(node.neighbors)==0]
    

    def neighbor_matrix(self, connection_range=None, weighted=False):
        """
        Function to compute the neighbor matrix of the swarm. 
        The neighbor matrix is a 2D matrix where each entry [i][j] is equal to the edge weight if node i is a neighbor of node j, and 0 otherwise.

        Args:
            connection_range (int, optional): the maximum distance between two nodes to establish a connection. Defaults to None, which means using the attribute of the Swarm object if none specified.

        Returns:
            list(list(int)): the 2D matrix representing the neighbor relationships between the nodes in the swarm.
        """
        matrix = []
        if not connection_range:
            connection_range = self.connection_range  # Use the attribute of the Swarm object if none specified
        for n1 in self.nodes:
            matrix.append([n1.is_neighbor(n2, connection_range, weighted) for n2 in self.nodes])
        return matrix
        
    
    def remove_expensive_edges(self):
        """
        Function to remove expensive edges from the swarm.
        An edge is considered expensive if its weight is higher than the weighted shortest path length between its two nodes, 
        or if there are other shortest paths besides the direct edge.

        Returns:
            None: this function modifies the neighbor lists of the nodes in the swarm.
        """
        for node in self.nodes:
            n1 = node.id
            ncopy = dict(node.neighbors)
            for n2,w in ncopy.items():
                if nx.shortest_path_length(self.graph, n1, n2, weight='cost') < w:
                    node.remove_neighbor(n2)
                    if self.graph.has_edge(n1, n2):
                        self.graph.remove_edge(n1, n2)
                elif len(list(nx.all_shortest_paths(self.graph, n1, n2, weight='cost'))) > 1:
                    node.remove_neighbor(n2)
                    if self.graph.has_edge(n1, n2):
                        self.graph.remove_edge(n1, n2)
        
        
        
    def remove_node(self, node:Node):
        """
        Function to remove a node from the swarm unless it is already out.

        Args:
            node (Node): the node to remove.
        """
        if node in self.nodes:
            self.nodes.remove(node)
            if self.graph != None:
                self.graph.remove_node(node.id)
                
        
    def reset_connection(self):
        """
        Function to empty the neighbor dict of each node in the swarm.
        """
        for node in self.nodes:
            node.neighbors = {}
        if self.graph != None:
            self.graph.remove_edges_from(list(self.graph.edges.keys()))
                
            
    def reset_groups(self):
        """
        Function to reset the group ID to -1 for each node of the swarm.
        """
        for node in self.nodes:
            node.set_group(-1)
        if self.graph != None:
            nx.set_node_attributes(self.graph, values=-1, name='group')
 
    
    def create_graph(self):
        G = nx.Graph()
        G.add_nodes_from([(n.id, {'group':n.group}) for n in self.nodes])
        visited = []
        for node in self.nodes:
            n1 = node.id
            for n2, w in node.neighbors.items():
                if n1 != n2 and set((n1, n2)) not in visited:
                    visited.append((n1,n2))
                    G.add_edge(n1, n2, cost=w, proba=1/w)
        self.graph = G
    
    #*************** Metrics ******************
    def cluster_coef(self, weight=None):
        """
        Calculate the clustering coefficient of the graph.

        The clustering coefficient is a measure of the degree to which nodes in a graph tend to cluster together.
        It is a measure of the local density of connections in the graph.

        Parameters:
        weight (str, optional): The edge attribute that holds the numerical value used as a weight.
            If None, every edge has weight 1.

        Returns:
        dict: A dictionary where keys are nodes and values are the clustering coefficients of the nodes.

        Raises:
        NetworkXError: If the graph is not undirected.

        References:
        - Saramaki, J. et al (2008). Generalizations of the clustering coefficient to weighted complex networks.
        """
        return nx.clustering(self.graph, self.graph.nodes(), weight=weight)
    
    def connected_components(self):
        """
        Function to define the connected components in the network.

        Returns:
            list(list(int)): nested list of node IDs for each connected component.
        """
        visited = {}
        for node in self.nodes: # Initialize all nodes as unvisited
            visited[node.id] = False
        cc = []
        for node in self.nodes:
            if visited[node.id]==False: # Perform DFS on each unvisited node
                temp = []
                cc.append(self.DFSUtil(temp, node, visited))
        return cc
    
    def degree(self):
        """
        Function to compute the degree (aka the number of neighbors) of each node within the swarm.

        Returns:
            list(int): the list of node degrees.
        """
        return [node.degree() for node in self.nodes]   
    
    def DFSUtil(self, temp, node, visited):
        """
        Function to perform a Depth-First Search on the graph. Usually used to define all connected components in the swarm.

        Args:
            temp (list(int)): the list of visited node IDs so far.
            node (Node): the node to be analysed.
            visited (dict(int:bool)): the dictionary of matches between the node ID and its state (visited or not).

        Returns:
            list(int): the updated temp list.
        """
        visited[node.id] = True # Mark the current node as visited
        temp.append(node.id) # Store the vertex to list
        for n in node.neighbors:
            if n in self.nodes:
                if visited[n.id] == False: # Perform DFS on unvisited nodes
                    temp = self.DFSUtil(temp, n, visited)
        return temp
    
    def diameter(self, group):
        """
        Function to compute the diameter of the swarm. The swarm is first converted into a nx.Graph object (see help(Swarm.swarm_to_nxgraph)).
        The diameter of the swarm is defined as the maximum shortest path distance between all pairs of nodes, in terms of number of hops.

        Args:
            group (Swarm): the list of nodes to take into account.

        Returns:
            tuple: the diameter of the swarm as (source_id, target_id, diameter).
        """
        G = self.swarm_to_nxgraph()
        node_ids = [n.id for n in group.nodes]
        max_length = (0,0,0) # Source, target, path_length
        for ni in node_ids:
            for nj in node_ids:
                if nx.has_path(G, ni, nj):
                    spl = nx.shortest_path_length(G, ni, nj)
                    if spl > max_length[2]:
                        max_length = (ni, nj, spl)
        return max_length
    
    def graph_density(self):
        """
        Function to compute the graph density of the swarm. The graph density is defined as the ratio between the number of edges and the maximum
        possible number of such edges.
        Let N be the number of nodes in the swarm. Then the maximum number of edges max_edges = N*(N-1)/2.
        Let m be the number of existing edges in the swarm. Then the graph density GD = (2*m)/(N*(N-1)).

        Returns:
            float: the graph density between 0 and 1.
        """
        N = len(self.nodes)
        max_edges = N*(N-1)/2
        if max_edges == 0:
            return 0
        edges = 0
        for n in self.nodes:
            common_nodes = set(n.neighbors).intersection(self.nodes)
            edges += len(common_nodes)
        return edges/(2*max_edges) # Divide by 2 because each edge is counted twice
    
    def k_vicinity(self, depth=1):
        """
        Function to compute the k-vicinity (aka the extended neighborhood) of each node in the swarm.
        The k-vicinity corresponds to the number of direct and undirect neighbors within at most k hops from the node.

        Args:
            depth (int, optional): the number of hops for extension. Defaults to 1.

        Returns:
            list(int): list of k-vicinity values for each node.
        """
        return [node.k_vicinity(depth) for node in self.nodes]
    
    def shortest_paths_lengths(self, group):
        """
        Function to compute all the shortest paths between each pair of nodes (Dijkstra algorithm) and return their length. The swarm is 
        first converted into a nx.Graph object (see help(Swarm.swarm_to_nxgraph)).

        Args:
            group (Swarm): the list of nodes to take into account.

        Returns:
            list(int): the list of the shortest path lengths.
        """
        G = self.swarm_to_nxgraph()
        node_ids = [n.id for n in group.nodes]
        lengths = []
        for ni in node_ids:
            for nj in node_ids:
                if nx.has_path(G, ni, nj) and nj != ni:
                    lengths.append(nx.shortest_path_length(G, source=ni, target=nj))
        return lengths 
    
    
    def strength(self):
        """
        Calculate the strength of each node in the swarm.
    
        The strength of a node is defined as the sum of the reciprocal of the costs (aka weights) of the edges connected to the node.
        This measure is used to quantify the importance or influence of a node in a network.
    
        Parameters:
        self (Swarm): The instance of the Swarm class.
    
        Returns:
        list(float): A list of the strength values for each node in the swarm.
        """
        return [sum([1/cost for cost in node.neighbors.values()]) for node in self.nodes]
    
    
    #************** Sampling algorithms ****************
        
    def FFD(self, n=10, p=0.7, s=1, by_id=False):
        """
        Function to perform graph sampling by the Forest Fire algorithm. 
        In the initial phase, n nodes are selected as "fire sources". Then, the fire spreads to the neighbors with a probability of p.
        We finally obtain n samples defined as the nodes burned by each source.

        Args:
            n (int, optional): the initial number of sources. Defaults to 10.
            p (float, optional): the fire spreading probability. Defaults to 0.7.
            s (int, optional): the random seed. Defaults to 1.
            by_id (bool, optional): if True, returns groups by node ids. Defaults to False.

        Returns:
            dict(int:list): the dictionary of group IDs and their corresponding list of node IDs.
        """
        sources = sample(self.nodes, n) # Initial random sources
        groups = {} # Dict(group ID:list(nodes))
        for group_id,src in enumerate(sources): # Initialize swarms
            src.set_group(group_id)
            groups[group_id] = [src]
        free_nodes = [n for n in self.nodes if n.group==-1]
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
                    nodes = [self.random_jump(s)] # If no neighbor, perform random jump in the graph
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
    
    def MIRW(self, n=10, s=1, by_id=False):
        """
        Function to perform graph division by the Multiple Independent Random Walk algorithm.
        In the initial phase, n nodes are selected as sources. Then they all perform random walks in parallel (see help(Node.random_walk) for
        more information). 
        We finally obtain n groups defined as the random walks from each source.

        Args:
            n (int, optional): the initial number of sources. Defaults to 10.
            s (int, optional): the random seed. Defaults to 1.
            overlap (bool, optional): if True, node groups are allowed to overlap. Defaults to False.

        Returns:
            dict(int:Swarm): the dictionary of group IDs and their corresponding Swarm group.
        """
        sources = sample(self.nodes, n) # Initial random sources
        groups = {} # Dict(group ID:list(Node))
        for group_id, src in enumerate(sources): # Initialize swarms
            src.set_group(group_id)
            groups[group_id] = [src]
        free_nodes = [n for n in self.nodes if n.group==-1]
        while free_nodes: # Spread paths
            for group_id in groups.keys():
                n_i = groups[group_id][-1] # Current node
                free_neighbors = set(free_nodes).intersection(n_i.neighbors)
                if free_neighbors: # At least one unassigned neighbor
                    n_j = n_i.random_walk(s*group_id) # Next node
                else:
                    n_j = self.random_jump(s) # If no neighbor, perform random jump in the graph
                n_j.set_group(n_i.group)
                groups[group_id].append(n_j)
                free_nodes.remove(n_j)
        if by_id:
            groups_by_id = {}
            for gid, nodes in groups.items():
                groups_by_id[gid] = [n.id for n in nodes]
            return groups_by_id
        return groups
    
    def RND(self, n=10, s=1, by_id=False):
        """
        Function to perform graph division by the Random Node Division algorithm.
        Each node choses a random group ID from the range given as parameter.

        Args:
            n (int, optional): the number of groups. Defaults to 10.
            s (int, optional): the random seed. Defaults to 1.

        Returns:
            dict(int:Swarm): the dictionary of group IDs and their corresponding Swarm group.
        """
        groups = {} # Dict(group ID:list(nodes))
        for i, node in enumerate(self.nodes):
            node.random_group(range(n), s*i)
        for group_id in range(n):
            groups[group_id] = [node for node in self.nodes if node.group==group_id]
        if by_id:
            groups_by_id = {}
            for gid, nodes in groups.items():
                groups_by_id[gid] = [node.id for node in nodes]
            return groups_by_id
        return groups
            
    def random_jump(self, s=1, overlap=False):
        """
        Function to choose a new node in the graph by performing a random jump.

        Args:
            s (int, optional): the random seed. Defaults to 1.
            overlap (bool, optional): if True, node groups are allowed to overlap. Defaults to False.

        Returns:
            Node: the randomly chosen node.
        """
        seed(s)
        search_list = self.nodes
        if not overlap: # Restrain the search list to unassigned nodes
            search_list = [n for n in self.nodes if n.group==-1]
        return choice(search_list)
    

    #************** Plot functions **************
    def plot_nodes(self, n_color='blue'):
        """
        Function to create a 3D-plot of the swarm at a given timestamp. 

        Args:
            n_color (str, optional) : Nodes color. Defaults to 'blue'.
        """
        fig = plt.figure(figsize=(8,8))
        ax = plt.axes(projection='3d')
        x_data = [node.x for node in self.nodes]
        y_data = [node.y for node in self.nodes]
        z_data = [node.z for node in self.nodes]
        ax.scatter(x_data, y_data, z_data, c=n_color, s=50)
    
    def plot_edges(self, figsize=(5,5), n_color='gray', edgecolors=None):
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection='3d')
        ax.set_axis_off()
        x_data = [node.x for node in self.nodes]
        y_data = [node.y for node in self.nodes]
        z_data = [node.z for node in self.nodes]
        ax.scatter(x_data, y_data, z_data, c=n_color, edgecolor='black', s=50)
        if edgecolors==None:
            edgecolors = {1:'blue', 4:'red'}
        for n1 in self.nodes:
            for nid,w in n1.neighbors.items():
                n2 = self.get_node_by_id(nid)
                ax.plot([n1.x, n2.x], [n1.y, n2.y], [n1.z, n2.z], c=edgecolors[w])
                    
                    
    
    

        