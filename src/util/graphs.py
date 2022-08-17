# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:16:03 2021

@author: John Meluso
"""

# Import libraries
from functools import reduce
from itertools import product as pd, combinations
import networkx as nx
import numpy as np


### Graph Parameter Setters ################################################

def set_graphs(graphs):
    '''Set the specified graphs to their default options.'''
    
    return_graphs = {}
    
    for graph in graphs:
        return_graphs[graph] = eval('set_' + graph + '()')
    
    return return_graphs


def set_all_graphs_default():
    '''Sets all graphs to their default options.'''
    
    return_graphs = {
        'complete': set_complete(),
        'empty': set_empty(),
        'power': set_power(),
        'random': set_random(),
        'ring_cliques': set_ring_cliques(),
        'rook': set_rook(),
        'small_world': set_small_world(),
        'star': set_star(),
        'tree': set_tree(),
        'wheel': set_wheel(),
        'windmill': set_windmill()
        }
    
    return return_graphs


def set_complete():
    '''Set options for a complete graph.'''
    return set_graph_opts('complete')


def set_empty():
    '''Sets options for an empty graph.'''
    return set_graph_opts('empty')


def set_power(m=[2],p=(0,1,3)):
    '''Set options for a power law graph.'''
    return set_graph_opts('power', m=m, p=p)


def set_random(p=(0,1,3)):
    '''Set options for a random graph.'''
    return set_graph_opts('random', p=p)


def set_ring_cliques():
    '''Set options for a ring of cliques graph.'''
    return set_graph_opts('ring_cliques')


def set_rook():
    '''Set options for a rook's graph.'''
    return set_graph_opts('rook')


def set_small_world(k=[2],p=(0,1,3)):
    '''Set options for a small world graph.'''
    return set_graph_opts('small_world',k=k,p=p)


def set_star():
    '''Set options for a star graph.'''
    return set_graph_opts('star')


def set_tree():
    '''Set options for a random tree graph.'''
    return set_graph_opts('tree')


def set_wheel():
    '''Set options for a wheel graph.'''
    return set_graph_opts('wheel')


def set_windmill():
    '''Set options for a windmill graph.'''
    return set_graph_opts('windmill')


def set_graph_opts(graph_type, **kwargs):
    '''Sets options for a specified graph type.'''
    
    opts = {}
    
    # Get allowed keys
    allowed_keys = get_allowed_keys(graph_type)
    
    # Update attributes
    for key, value in kwargs.items():
        if key in allowed_keys:
            if key == 'p':
                if isinstance(value,float) == 1:
                    opts[key] = np.array([value])
                elif isinstance(value, tuple) and len(value) == 3:
                    opts[key] = np.round(np.linspace(
                        value[0],value[1],value[2]),decimals=2)
                elif isinstance(value, list):
                    opts[key] = np.round(value,decimals=2)
                else: raise RuntimeError(f'Error in set_{graph_type}.')
            else:
                opts[key] = value
        else:
            raise RuntimeError(f'Input {key} invalid for {graph_type}.')
    
    return opts


def get_allowed_keys(graph_type):
    '''Gets the allowed keys for the graph type.'''
    
    allowed = {
        'complete': None,
        'empty': None,
        'power': {'m','p'},
        'random': {'p'},
        'ring_cliques': None,
        'rook': None,
        'small_world': {'k','p'},
        'star': None,
        'tree': None,
        'wheel': None,
        'windmill': None
        }
    
    return allowed[graph_type]


def product_dict(**kwargs):
    '''Iterate over entries in each value for each key.'''
    
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in pd(*vals):
        yield dict(zip(keys, instance))
        

### Graph Getters ##########################################################

def get_complete(team_size, **team_graph_opts):
    '''Returns an instance of a complete graph.'''
    return nx.complete_graph(team_size)

def get_empty(team_size, **team_graph_opts):
    '''Returns an instance of an empty graph.'''
    return nx.empty_graph(team_size)

def get_power(team_size, **team_graph_opts):
    '''Returns an instance of a power law graph.'''
    if team_graph_opts == {}:
        team_graph_opts['m'] = 2
        team_graph_opts['p'] = 0.3
    return nx.powerlaw_cluster_graph(team_size,**team_graph_opts)

def get_random(team_size, **team_graph_opts):
    '''Returns an instance of a random graph.'''
    if team_graph_opts == {}:
        team_graph_opts['p'] = 0.3
    return nx.gnp_random_graph(team_size,**team_graph_opts)

def get_ring_cliques(team_size, **team_graph_opts):
    '''Returns an instance of a ring of cliques graph.'''
    cliques = int(np.round(np.sqrt(team_size), decimals=0))
    sizes = cliques
    return nx.ring_of_cliques(cliques, sizes)

def get_rook(team_size, **team_graph_opts):
    '''Returns an instance of a rook's graph.'''
    side_length = int(np.round(np.sqrt(team_size), decimals=0))
    board_size = (side_length, side_length)
    return gen_rook(board_size)

def get_small_world(team_size, **team_graph_opts):
    '''Returns an instance of a small world graph.'''
    if team_graph_opts == ():
        team_graph_opts['k'] = 2
        team_graph_opts['p'] = 0
    return nx.watts_strogatz_graph(team_size,**team_graph_opts)

def get_star(team_size, **team_graph_opts):
    '''Returns an instance of a star graph. NetworkX takes n outer nodes, so
    subtract 1'''
    return nx.star_graph(team_size - 1)

def get_tree(team_size, **team_graph_opts):
    '''Returns an instance of a random tree graph.'''
    return nx.random_tree(team_size)

def get_wheel(team_size, **team_graph_opts):
    '''Returns an instance of a wheel graph.'''
    return nx.wheel_graph(team_size)

def get_windmill(team_size, **team_graph_opts):
    '''Returns an instance of a windmill graph.'''
    sqrt = int(np.round(np.sqrt(team_size),decimals=0))
    team_graph_opts['n'] = sqrt + 1
    team_graph_opts['k'] = sqrt
    return nx.windmill_graph(**team_graph_opts)


### Graph Generators #######################################################

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