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
        self.messages = [] # List of messages received from sources. List of Packet objects
        
    def __str__(self):
        """
        Node object descriptor
        
        Returns:
            str: a string description of the node
        """
        nb_neigh = len(self.neighbors)
        return f"Node ID {self.id} ({self.x},{self.y},{self.z}) has {nb_neigh} neighbor(s)"
    
    def show_messages(self):
        nb_sig = len([m for m in self.messages if m.tos==0])
        nb_data = len([m for m in self.messages if m.tos==1])
        print(f'Node {self.id} received {nb_sig} signaling messages and {nb_data} data messages.')
        for m in self.messages:
            print(m)
    
    def add_neighbor(self, node):
        """
        Function to add a node to the neighbor list of the node unless it is already in its list.
        
        Args:
            node (int): the node to add
        """
        if node not in self.neighbors:
            self.neighbors.append(node)
        
    def remove_neighbor(self, node):
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
    
    def k_vicinity(self, depth=1):
        """
        Function to compute the k-vicinity (aka the extended neighborhood) of the node.

        Args:
            depth (int, optional): the number of hops for extension. Defaults to 1.

        Returns:
            int: the length of the extended neighbor list of the node
        """
        kv = self.neighbors.copy()
        for i in range(depth-1):
            nodes = kv
            kv.extend([n for node in nodes for n in node.neighbors])
        return len(set(kv))
    
    def compute_dist(self, node):
        """
        Function to compute the Euclidean distance between two nodes.
        
        Args:
            node (Node): the node to compute the distance with

        Returns:
            float: the Euclidean distance between the two nodes
        """
        return dist((self.x, self.y, self.z) , (node.x, node.y, node.z))
    
    def is_neighbor(self, node, connection_range=0):
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
    
    def cluster_coef(self):
        """
        Function to compute the clustering coefficient of a node, which is defined as
        the existing number of edges between the neighbors of a node divided by the maximum
        possible number of such edges.

        Returns:
            float: the clustering coefficient of the node between 0 and 1
        """
        dv = self.degree()
        max_edges = dv*(dv-1)/2
        if max_edges == 0:
            return 0
        edges = 0
        for v in self.neighbors:
            common_elem = set(v.neighbors).intersection(self.neighbors)
            edges += len(common_elem)
        return edges/(2*max_edges)
    
    def receive(self, pkt):
        self.state = 1
        self.messages.append(pkt)
    
    def epidemic(self, pkt):
        """
        Function to simulate an epidemic propagation from the node to all its neighbors.
        The function affects the state attribute of the Node objects:
            0: the node has not received any message yet
            1: the node is currently bearing a message
            -1: the node has transmitted its message
        """
        transmissions = []
        for node in self.neighbors:
            if node.state != -1: # Previous state: the node has not transmitted the message yet
                node.receive(pkt)
                transmissions.append(node.id)
        nb_transmissions = len(transmissions)
        if nb_transmissions > 0: 
            self.state = -1 # the node has transmitted the message to at least one neighbor. Else, keeps it till next connection
            print(f'\nNode {self.id} has transmitted {nb_transmissions} message(s) to:')
            print([t for t in transmissions])
        else:
            print(f'\nNode {self.id} has no one to transmit to (state {self.state}).\n')
            
    def listen_and_retransmit(self):
        """
        Function to simulate a retransmission in an epidemic propagation from the node to all its neighbors.
        The function is always called by a (-1) node and affects the state of (0) neighbor nodes.
        """
        retransmissions = []
        for node in self.neighbors:
            if node.state == 0: # Previous state: the node has not received the message yet
                node.state = 1 # New state: the node receives a (new) message 
                node.copies_from.append(self.id)
                retransmissions.append(node.id)
        nb_transmissions = len(retransmissions)
        if nb_transmissions > 0: 
            print(f'\nNode {self.id} has retransmitted {nb_transmissions} message(s) (state {self.state}):')
            print([t for t in retransmissions])

    
    
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
    
    def k_vicinity(self, depth=1):
        """
        Function to compute the k-vicinity (aka the extended neighborhood) of each node in the swarm.

        Args:
            depth (int, optional): the number of hops for extension. Defaults to 1.

        Returns:
            list: list of k-vicinity values (int)
        """
        return [node.k_vicinity(depth) for node in self.nodes]
    
    def cluster_coef(self):
        """
        Function to compute the clustering coefficient distribution of the swarm.

        Returns:
            list: list of clustering coefficient values (float)
        """
        return [node.cluster_coef() for node in self.nodes]
    
    def DFSUtil(self, temp, node, visited):
        visited[node.id] = True
        temp.append(node.id) # Store the vertex to list
        for n in node.neighbors:
            if visited[n.id] == False:
                temp = self.DFSUtil(temp, n, visited)
        return temp
    
    def connected_components(self):
        visited = [False]*len(self.nodes)
        cc = []
        for node in self.nodes:
            if visited[node.id]==False:
                temp = []
                cc.append(self.DFSUtil(temp, node, visited))
        print('Number of connected components:', len(cc))
        return cc
        
    
    def distance_matrix(self):
        """
        Function to compute the Euclidean distance matrix of the swarm.

        Returns:
            list: a 2-dimensional distance matrix formatted as matrix[node1][node2] = distance
        """
        matrix = []
        for n1 in self.nodes:
            matrix.append([n1.compute_dist(n2) for n2 in self.nodes if n1.id != n2.id])
        return matrix
    
    def neighbor_matrix(self, connection_range=None):
        """
        Function to compute the neighbor matrix of the swarm.
        If two nodes are neighbors (according to the given connection range), the row[col] equals 1. Else 0.

        Args:
            connection_range (int, optional): the connection range of the swarm. Defaults to None.

        Returns:
            list: the 2-dimensional neighbor matrix formatted as matrix[node1][node2]
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
    
    def reset_propagation(self):
        """
        Function to reset all nodes states to 0 for the simulation of a message propagation and erase message history.
        """
        for node in self.nodes:
            node.state = 0
            node.copies_from = []
            
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
    
    def get_swarm_msg(self):
        """
        Function to retrieve the current message copies of each node of the swarm as a dict object.

        Returns:
            dict: (key, value) is (node ID, Packet)
        """
        msg = {}
        for node in self.nodes:
            msg[node.id] = node.messages
        return msg
   
    def epidemic(self, prev_state = None, prev_msg=None, enhanced=False):
        """
        Function to simulate an epidemic message propagation in the swarm.

        Args:
            prev_state (dict, optional): the previous (or initial) state of the swarm. Defaults to None.
            prev_msg (dict, optional): the previous (or initial) messages of the swarm. Defaults to None.
        """
        bearers = []
        listeners = []
        for node_id,state in prev_state.items():
            node = self.get_node_by_id(node_id)
            node.state = state
            node.messages = prev_msg[node_id]
            if state == 1: # Message bearer
                bearers.append(node)
            elif state == -1:
                listeners.append(node)
        print(len(bearers), 'Bearer node(s):\n', [n.id for n in bearers])
        for node in bearers:
            node.epidemic() 
        if enhanced:
            for node in listeners:
                node.listen_and_retransmit()

        
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
    
    def plot_edges(self):
        fig = plt.figure(figsize=(12,12))
        ax = plt.axes(projection='3d')

        x_data = [node.x for node in self.nodes]
        y_data = [node.y for node in self.nodes]
        z_data = [node.z for node in self.nodes]
        ax.scatter(x_data, y_data, z_data, c='blue', s=50)

        for node in self.nodes:
            for n in node.neighbors:
                ax.plot([node.x, n.x], [node.y, n.y], [node.z, n.z], c='red')
        
#==============================================================================================

class Packet:
    """
    Packet object, representing a Network layer message
    """
    
    def __init__(self, src_addr:int, tos:int, dest_addr=None, ttl=64, data=None):
        """
        Packet object constructor

        Args:
            src_addr (int): network address or ID of the sender
            tos (int): Type of Service, 0 if signaling packet, 1 if data packet
            dest_addr (int, optional): network address or ID of the destination. If broadcast, defaults to None.
            ttl (int, optional): Time To Live. Defaults to 64.
            data (string, optional): information to share. Defaults to None.
        """
        self.id = (src_addr, dest_addr, tos) #Packet is identified by the sender & type of service
        self.src_addr = src_addr
        self.dest_addr = dest_addr
        self.tos = tos #Type of Service: 0 (SIGNALING) or 1 (DATA)
        self.ttl = ttl
        self.data = data
        
    def __str__(self):
        """
        Packet object descriptor

        Returns:
            str: the string description of the packet
        """
        dest = self.dest_addr
        if dest == None:
            dest = 'BROADCAST'
        ids = f'SRC: {self.src_addr}\t DEST: {dest}\t'
        tos = 'SIGNALING'
        if self.tos == 1:
            tos = 'DATA'
        desc = f'ToS: {tos}\tTTL: {self.ttl}\n'
        data = f'DATA: {self.data}\n'
        return ids+desc+data
    
    def is_packet(self, pkt):
        """
        Packet comparison

        Args:
            pkt (Packet): the packet to compare with

        Returns:
            Bool: True if packets are equal, else False
        """
        if pkt.id == self.id:
            return True 
        else:
            return False
        
    def update_ttl(self):
        self.ttl -= 1
            
        

class EpidemicPacket(Packet):
    
    def __init__(self, src_addr:int, ttl=64, data=None):
        Packet.__init__(self, src_addr=src_addr, tos=1, ttl=ttl, data=data) # Epidemic propagation only transmits DATA packets (not SIGNALING)
        self.protocol = 'EPIDEMIC'
        
    def __str__(self):
        return f'PROTOCOL: {self.protocol}\n' + Packet.__str__(self)
    
    