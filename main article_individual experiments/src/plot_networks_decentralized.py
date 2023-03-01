# -*- coding: utf-8 -*-
"""
Created on Tue Aug 8 12:32:49 2022

@author: John Meluso
"""

# Import libraries
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np

# Import source files
import fig_settings as fs
from classes.Team import get_graph
import util.graphs as ug


fs.set_fonts() 

def plot_decentralized_graphs():
    '''Plots all of the networks.'''
    
    # Set iterating params
    decent_graphs = {
        'empty_na_na_na':  dict(
            label = 'Empty graph ($k=0$)',
            graph_type = 'empty',
            graph_opts = ug.set_empty(),
            layout = 'circular',
            ),
        'small_world_2_na_0.0':  dict(
            label = 'Ring ($k=2$)',
            graph_type = 'small_world',
            graph_opts = ug.set_small_world(k=2,p=0.0),
            layout = 'circular',
            ),
        'small_world_4_na_0.0':  dict(
            label = 'Small world ($k=4$)',
            graph_type = 'small_world',
            graph_opts = ug.set_small_world(k=4,p=0.0),
            layout = 'circular',
            ),
        'rook_na_na_na':  dict(
            label = 'Rook\'s graph ($k=4$)',
            graph_type = 'rook',
            graph_opts = ug.set_rook(),
            layout = 'grid',
            ),
        'small_world_6_na_0.0':  dict(
            label = 'Small world ($k=6$)',
            graph_type = 'small_world',
            graph_opts = ug.set_small_world(k=6,p=0.0),
            layout = 'circular',
            ),
        'complete_na_na_na': dict(
            label = 'Complete graph ($k=8$)',
            graph_type = 'complete',
            graph_opts = ug.set_complete(),
            layout = 'circular',
            )
        }
    size = 9
    
    # Create figure
    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=fs.fig_size(0.5,0.25,2),
        dpi=1200,
        # tight_layout=True,
        constrained_layout=True,
        gridspec_kw={'hspace': 0.1, 'wspace': 0.2},
        )
    
    # Loop through all functions
    for ax, (graph, props) in zip(axs.flatten(), decent_graphs.items()):
            
        # Get graph options and name
        graph_type = props['graph_type']
        graph_opts = props['graph_opts']
        graph_layout = props['layout']
        graph_label = props['label']
            
        # Create the plot
        plot_network(ax, graph_type, size, graph_opts, graph_label, graph_layout)
    
    # Save figure
    fs.save_pub_fig('networks_decentralized')

def plot_network(ax, network_type, team_size, network_opts, network_label, 
                 network_layout):
    
    # Set the seed for everything but windmill, which can't accept a seed
    if network_type != 'windmill': network_opts['seed'] = 42
    
    # Get the graph
    graph = get_graph(network_type, team_size, **network_opts)
    scale = 1    
    size = 25
    nc = '#AA0000'
    ec = '#808080'
    width = 0.5
    
    # Draw with Kamada-Kawai layout
    if network_layout == 'kk':
        pos = nx.kamada_kawai_layout(graph, scale=scale)
        nx.draw_networkx(graph, pos=pos, ax=ax, with_labels=False,
                         node_size=size, node_color=nc, edge_color=ec,
                         width=width)
        
    # Draw with circular layout
    elif network_layout == 'circular':
        pos = nx.circular_layout(graph, scale=scale)
        nx.draw_networkx(graph, pos=pos, ax=ax, with_labels=False,
                         node_size=size, node_color=nc, edge_color=ec,
                         width=width)
        
    # Draw with grid layout
    elif network_layout == 'grid':
        side = np.round(np.sqrt(team_size), decimals=0)
        xx, yy = np.meshgrid(range(int(side)),range(int(side)),indexing='xy')
        pos = {ii: (x,-y) for ii, (x,y) in enumerate(zip(xx.flat,yy.flat))}
        edges = nx.to_edgelist(graph)
        edge_graph = nx.DiGraph(edges)
        nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_size=size,
                               node_color=nc)
        draw_curved_edges(ax, edge_graph, pos, ec, width)
            
    else: raise RuntimeError(f'{network_type} is not a valid network.')
    
    # Draw the network
    ax.set_aspect(1)
    
    # Set labels
    ax.set_xlabel(network_label)
    for spine in ax.spines.values():
        spine.set_visible(False)

def draw_curved_edges(ax, graph, pos, edge_color, linewidth):
    
    # Loop through edges in graph
    for (n1,n2) in graph.edges():
        
        # Get node positions
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        
        # Add lines between each pair of nodes
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle='-',
            shrinkA=0,
            shrinkB=0,
            color=edge_color,
            linewidth=linewidth,
            connectionstyle='Arc3,rad=-0.2',
            linestyle='-',
            zorder=1, # arrows go behind nodes
            )  
        ax.add_patch(arrow)


if __name__ == '__main__':
    plot_decentralized_graphs()
