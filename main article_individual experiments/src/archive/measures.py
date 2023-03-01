# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:16:07 2022

@author: John Meluso
"""

import networkx as nx
import networkx.algorithms.assortativity as assort
import networkx.algorithms.centrality as cent
import numpy as np
import pandas as pd
from statistics import mean, stdev
from functools import reduce
from itertools import combinations

def gen_rook(board_size):
    
    # Construct numpy array of nodes
    node_array = np.array(np.arange(
        reduce(lambda x,y: x*y, board_size)
        ).reshape(board_size))
    
    # Create iterable of indices in node_array
    node_iter = np.nditer(node_array, flags=['multi_index'])

    # Create list & dictionary of indices
    node2location = []
    location2node = {}
    for xx in node_iter:
        curr_index = node_iter.multi_index
        loc = str(curr_index)
        location2node[loc] = int(xx)
        node2location.append(curr_index)
    
    # Create list of all combinations in N-dim board
    combo_list = list(combinations(node2location,2))
    
    # Initialize list of edges
    edge_indices = []
    edge_locations = []
    
    # Iterate through all combos
    for combo in combo_list:
        
        # Select elements that share every index but one
        # (same row, diff col, etc.)
        if np.sum(np.equal(combo[0],combo[1])) == len(board_size)-1:
            index = (location2node[str(combo[0])],location2node[str(combo[1])])
            edge_indices.append(index)
            edge_locations.append(combo)
        
    # Create graph instances
    graph_builder = nx.Graph()
    graph = nx.Graph()
    
    # Add edges to graph, although they will not be ordered
    graph_builder.add_edges_from(edge_indices)
    
    # Sort the nodes to build a new ordered graph
    graph.add_nodes_from(sorted(graph_builder.nodes(data=True)))
    graph.add_edges_from(graph_builder.edges(data=True))
    
    # Return graph
    return graph

n=25
cliques = int(np.round(np.sqrt(n), decimals=0))
side_length = int(np.round(np.sqrt(n), decimals=0))
board_size = (side_length, side_length)
nets = {
    'complete': nx.complete_graph(n),
    'empty': nx.empty_graph(n),
    'power': nx.powerlaw_cluster_graph(n, m=2, p=0.3),
    'random': nx.gnp_random_graph(n, p=0.3),
    'ring': nx.watts_strogatz_graph(n, 2, 0),
    'ring_cliques': nx.ring_of_cliques(cliques, cliques),
    'rook': gen_rook(board_size),
    'star': nx.star_graph(n),
    'tree': nx.random_tree(n),
    'wheel': nx.wheel_graph(n),
    }

measures = {'eig_mean':[], 'eig_std':[], 'knn_mean': [], 'knn_std': []}
names = []

for name, net in nets.items():
    eig = cent.eigenvector_centrality(net,tol=1e-3,max_iter=1000).values()
    knn = [knn/(len(net) - 1) for knn
             in assort.average_neighbor_degree(net).values()]
    names.append(name)
    measures['eig_mean'].append(mean(eig))
    measures['eig_std'].append(stdev(eig))
    measures['knn_mean'].append(mean(knn))
    measures['knn_std'].append(stdev(knn))
    
measures = pd.DataFrame(measures, index=names)