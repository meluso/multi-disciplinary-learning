# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 16:44:54 2022

@author: John Meluso
"""

# Import libraries
import itertools as it
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import seaborn as sns

# Import source files
import classes.Objective as Objective
import fig_settings as fs
from classes.Team import get_objective, get_graph
import util.functions as uf
import util.graphs as ug


fs.set_fonts()

def plot_examples(subfig):
    
    # Specify example tasks
    tasks = {
        'average': {
            'opts': uf.set_average(wts=['node']),
            'label': '(a) Average'
            },
        'ackley': {
            'opts': uf.set_ackley(),
            'label': '(b) Ackley'
            },
        'losqr_hiroot': {
            'opts': uf.set_losqr_hiroot(wts=['node'], exp=['degree']),
            'label': '(c) Low Square-High Root'
            },
        'kth_root': {
            'opts': uf.set_kth_root(wts=['node']),
            'label': '(d) $K+1$ Root'
            },
        'kth_power': {
            'opts': uf.set_kth_power(wts=['node']),
            'label': '(e) $K+1$ Power'
            }
        }
    
    # Specify example networks
    networks = {
        'complete': {
            'opts': ug.set_complete(),
            'label': '(f) Complete Graph'
            },
        'empty': {
            'opts': ug.set_empty(),
            'label': '(g) Empty Graph'
            },
        'power': {
            'opts': ug.set_power(m=[2],p=0.5),
            'label': '(h) Pref. Attachment\n(2 edges, 0.5 triangle prob.)'
            },
        'random': {
            'opts': ug.set_random(p=0.5),
            'label': '(i) Random Graph\n(0.5 edge prob.)'
            },
        'small_world': {
            'opts': ug.set_small_world(k=[2],p=0.5),
            'label': '(j) Small World\n(0.5 edge move prob.)'
            }
        }
    
    # Create figure, first row for tasks, second for networks
    sub2figs = subfig.subfigures(2, 1, hspace=0)
    
    # Get axes
    axsTasks = sub2figs[0].subplots(
        nrows=1,
        ncols=len(tasks),
        gridspec_kw=dict(
            wspace=0.1
            ),
        subplot_kw={'projection': '3d'}
        )
    axsNets = sub2figs[1].subplots(
        nrows=1,
        ncols=len(networks),
        gridspec_kw=dict(
            wspace=0.1
            )
        )
    
    # Create tasks
    for tt, (task, values) in enumerate(tasks.items()):
        task_opts = values['opts']
        label = values['label']
        for opts in list(it.product(*task_opts.values())):
            plot_task(axsTasks[tt], task, opts, label) 
        
    # Create networks
    for nn, (network, values) in enumerate(networks.items()):
        opts = {key: value[0] for key, value in values['opts'].items()}
        label = values['label']
        plot_network(axsNets[nn], network, opts, label)
        
    return subfig
    

def plot_task(ax, fn_type, fn_opts, fn_label):
    
    # Get fn opts
    opts = uf.get_fn_opts(fn_type, fn_opts)
    
    # Set some constants
    x_min = 0
    x_max = 1
    ndivs = 100
    ks = [2,10]
    
    # Set x and y ranges
    x_range = np.linspace(x_min,x_max,ndivs)
    y_range = np.linspace(x_min,x_max,ndivs)
    x_mesh, y_mesh = np.meshgrid(x_range,y_range)
    z_mesh = np.round(np.zeros((ndivs,ndivs)),decimals=4)
    
    # Create objective
    objective = get_objective(fn_type, opts, ks)
    
    # Update zmesh
    for (ii,xx), (jj,yy) \
        in it.product(enumerate(x_range),enumerate(y_range)):
        
        z_mesh[ii,jj] = objective([xx,yy])
        
    # Plot the zmesh
    cmap = sns.color_palette('rocket', as_cmap=True)
    newcmp = mcolors.ListedColormap(cmap(np.linspace(0.05, 0.95, 256)))
    surf = ax.plot_surface(x_mesh,y_mesh,z_mesh,cmap=newcmp,
                           rcount=ndivs, ccount=ndivs)
    ax.tick_params(pad=-3)
    ax.set_xlabel('$x_1$', labelpad=-10)
    ax.set_ylabel('$x_2$', labelpad=-10)
    ax.text(x=0.85, y=1, z=1.2, s='$g(x_1,x_2)$', fontsize=6)
    ax.set_zticks(
        ticks=[0.0, 0.5, 1.0]
        )
    ax.margins(tight=True)
    ax.set_title(fn_label, pad=-3)
    
    return ax

def plot_network(ax, network_type, network_opts, network_label):
    
    team_size = 9
    network_opts['seed'] = 42
    graph = get_graph(network_type, team_size, **network_opts)
    pos = nx.rescale_layout_dict(nx.circular_layout(graph), scale=0.6)
    nx.draw_networkx(graph, pos=pos, ax=ax, with_labels=False, node_size=25,
                     node_color='#AA0000', edge_color='#808080')
    ax.margins(tight=True)
    ax.set_title(network_label, y=-0.2)
    ax.set_aspect(1)
    ax.set_axis_off()
    
def test_plot_examples():
    
    fig = plt.figure(
        figsize=(12,5.5),
        dpi=1200,
        layout='constrained',
        )
    subfigs = fig.subfigures()
    subfigs = plot_examples(subfigs)
    fig.show()
    
    

if __name__ == '__main__':
    import plot_tasks_networks_difficulties as ptnd
    ptnd.plot_tasks_networks_difficulties()