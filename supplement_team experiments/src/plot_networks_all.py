# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:32:49 2022

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
import util.writing as uw


fs.set_fonts() 

def plot_all_graphs():
    '''Plots all of the networks.'''
    
    # Set iterating params
    all_graphs = uw.get_graphs()
    sizes = [4, 9, 16, 25]
    
    # Create figure
    fig, axs = plt.subplots(
        nrows=len(all_graphs),
        ncols=len(sizes),
        figsize=fs.fig_size(0.85,0.9),
        dpi=1200,
        # tight_layout=True,
        constrained_layout=True,
        gridspec_kw={'wspace': 0.5},
        )
    
    # Loop through all functions
    for gg, (graph, props) in enumerate(all_graphs.items()):
        for ss, size in enumerate(sizes):
            
            # Get graph options and name
            graph_type = props['graph_type']
            graph_opts = props['graph_opts']
            graph_layout = props['layout']
            graph_label = props['label']
                
            # Create the plot
            plot_network(axs[gg,ss], graph_type, size, graph_opts, graph_label,
                         graph_layout)
    
    # Just keep outer labels
    for ax in axs.flat: ax.label_outer()
    
    # Save figure
    fs.save_pub_fig('networks_all')

def plot_network(ax, network_type, team_size, network_opts, network_label, 
                 network_layout):
    
    # Set the seed for everything but windmill, which can't accept a seed
    if network_type != 'windmill': network_opts['seed'] = 42
    
    # Get the graph
    graph = get_graph(network_type, team_size, **network_opts)
    scale = 1    
    
    # Draw with Kamada-Kawai layout
    if network_layout == 'kk':
        pos = nx.kamada_kawai_layout(graph, scale=scale)
        nx.draw_networkx(graph, pos=pos, ax=ax, with_labels=False, node_size=2,
                         node_color='#AA0000', edge_color='#808080', width=0.5)
        
    # Draw with circular layout
    elif network_layout == 'circular':
        pos = nx.circular_layout(graph, scale=scale)
        nx.draw_networkx(graph, pos=pos, ax=ax, with_labels=False, node_size=2,
                         node_color='#AA0000', edge_color='#808080', width=0.5)
        
    # Draw with grid layout
    elif network_layout == 'grid':
        side = np.round(np.sqrt(team_size), decimals=0)
        xx, yy = np.meshgrid(range(int(side)),range(int(side)),indexing='xy')
        pos = {ii: (x,-y) for ii, (x,y) in enumerate(zip(xx.flat,yy.flat))}
        edges = nx.to_edgelist(graph)
        edge_graph = nx.DiGraph(edges)
        nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_size=2,
                               node_color='#AA0000')
        draw_curved_edges(ax, edge_graph, pos)
            
    else: raise RuntimeError(f'{network_type} is not a valid network.')
    
    # Draw the network
    ax.set_aspect(1)
    
    # Set labels
    ax.set_xlabel(f'{team_size} Agents')
    ax.set_ylabel(network_label, rotation='horizontal', ha='right', va='bottom')
    for spine in ax.spines.values():
        spine.set_visible(False)

def draw_curved_edges(ax, graph, pos):
    
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
            color='#808080',
            linewidth=0.5,
            connectionstyle='Arc3,rad=-0.2',
            linestyle='-',
            zorder=1, # arrows go behind nodes
            )  
        ax.add_patch(arrow)


if __name__ == '__main__':
    plot_all_graphs()
