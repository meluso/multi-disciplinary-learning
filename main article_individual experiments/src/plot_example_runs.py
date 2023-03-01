# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:20:51 2022

@author: John Meluso
"""

# Import libraries
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

# Import source files
import fig_settings as fs
import util.data as ud
import util.plots as up
import util.variables as uv

All = slice(None)
fs.set_fonts()

#%% Supporting Functions

def get_data(execset):
    
    # Get execset data
    file_in = f'../data/sets/execset{execset:03}_stats.pickle'
    params2stats = ud.load_pickle(file_in)
    
    # Build row indices
    # name = ['team_performance','team_productivity']
    name = ['team_performance']
    stat = 'mean'
    cols = (name, stat)
    var2slice = get_var2slice()
    
    # Slice dataset down to the following fields:
    # Model | Graph | Team Fn | Nbhd Fn | name
    df = params2stats.loc[tuple(var2slice.values()),cols].reset_index()
    if stat is not None:
        df = df.droplevel(1,axis=1)
        
    # Build cumulative variables for fns and graphs
    var_prefixes = ['team_graph','team_fn','nbhd_fn','agent_fn']
    for prefix in var_prefixes:
        df = combine_columns(df, prefix)
        
    # Mask objectives
    df = mask_with_objectives(df)
    
    # Drop all the variables we don't need
    df = df[['team_graph','team_fn','agent_steplim','run_step',
             'team_performance',
             # 'team_productivity'
             ]]
    
    # Melt outcomes columns into variable and value columns
    df = df.melt(
        id_vars=['team_graph','team_fn','run_step','agent_steplim'],
        value_vars=[
            'team_performance',
            # 'team_productivity'
            ]
        )
    
    return df

def get_var2slice():
    
    # Build row indices
    var2slice = {key: value for key, value in uv.get_default_slices().items()}
    var2slice['model_type'] = ['3xx']
    var2slice['team_size'] = 9
    var2slice['agent_steplim'] = 0.1
    del var2slice['run_ind']
    
    return var2slice

def mask_with_objectives(df):
    
    # Build objective conditions
    mask = (df.team_fn == df.nbhd_fn) & (
        (df.team_fn == 'average_na_na_node')
        | (df.team_fn == 'ackley_na_na_na')
        | (df.team_fn == 'losqr_hiroot_degree_na_node')
        | (df.team_fn == 'kth_root_na_na_node')
        | (df.team_fn == 'kth_power_na_na_node')
        )
    
    # Get just rows with objectives to display
    df = df[mask]
    
    return df

def combine_columns(df, prefix):
    cols_with_prefix = df.columns[df.columns.str.startswith(prefix)]
    for col in cols_with_prefix:
        if prefix in df.columns:
            df[prefix] = [x + '_' + str(y) for x, y in zip(df[prefix], df[col])]
        else:
            df[prefix] = df[col]
    return df

def get_outcomes_and_descriptors():
    
    variables = []
    for out in uv.get_outcomes().keys(): variables.append(out)
    for desc in uv.get_descriptors().keys(): variables.append(desc)
    
    return variables

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

#%% Plotting Functions

def plot_example_run_averages(execset,save_fig=True,xscale=None,yscale=None):
    
    # Get data and unique values
    df = get_data(execset)
    
    # Set up figure properties
    fn2title = {
        'average_na_na_node': '(a) Average Function',
        'ackley_na_na_na': '(b) Ackley Function',
        'losqr_hiroot_degree_na_node': '(c) Low Square-High Root Fn.',
        'kth_root_na_na_node': '(d) $K+1$ Root Function',
        'kth_power_na_na_node': '(e) $K+1$ Power Function'
        }
    graphs = ['empty_na_na_na', 'small_world_2_na_0.5', 'power_na_2_0.5',
              'random_na_na_0.5', 'complete_na_na_na']
    graph_labels = up.get_graph_labels()
    graph2label = {graph: graph_labels[graph] for graph in graphs}
    var2ylabel = {
        'team_performance': 'Average Team Performance',
        # 'team_productivity': 'Average Agent Productivity'
        }
    xlabel = 'Time Step $t$'
    
    # Create style cycler
    cmap = get_continuous_cmap(
        ['#FFAAAA','#FF5555','#FF0000','#AA0000','#550000'])
    cmap = cmap(np.linspace(0,1,5))
    style_cycler = cycler(linestyle=['-','--','-.',(0, (5, 1)),'-'])
    color_cycler = cycler(color=cmap)
    joint_cycler = (style_cycler + color_cycler)
    plt.rc('axes', prop_cycle=joint_cycler)
    
    # Set up the figures
    fig, axs = plt.subplots(
        nrows=len(var2ylabel), ncols=len(fn2title),
        sharex=True, sharey='row', figsize=fs.fig_size(1, 0.22), dpi=1200
        )
    
    for xx, (fn, title) in enumerate(fn2title.items()):
        for yy, (var, ylabel) in enumerate(var2ylabel.items()):
            for zz, (graph, linelabel) in enumerate(graph2label.items()):
            
                # Subset the data by function, variable, and graph
                m1 = (df['team_fn'] == fn)
                m2 = (df['variable'] == var)
                m3 = (df['team_graph'] == graph)
                data = df[m1 & m2 & m3]
                    
                # Draw the lines
                axs[xx].plot(data['run_step'], data['value'], label=linelabel)
                if xscale is not None:
                    axs[xx].set_xscale(xscale['value'],base=xscale['base'])
                if yscale is not None:
                    axs[xx].set_yscale(yscale['value'],base=yscale['base'])
                
                
        # Add labels for the correct rows and columns
        if xx==0: axs[xx].set_ylabel(ylabel)
        if yy==0: axs[xx].set_title(f'{title}')
        if yy==0: axs[xx].set_xlabel(xlabel)
        axs[xx].grid(True)
        axs[xx].set_xlim(xmin=0, xmax=25)
        axs[xx].set_ylim(ymin=0, ymax=1)
        fs.set_border(axs[xx], bottom=True, left=True)
        handles, lables = axs[xx].get_legend_handles_labels()
            
    # Create legend
    fig.legend(handles, lables,
               loc='lower center',
               bbox_to_anchor=(0.5, 0),
               ncol=5)
    
    # Show the plots
    plt.tight_layout(rect=(0,0.08,1,1))
    if save_fig: fs.save_pub_fig('example_outcomes', kwargs={'bbox_inches': 'tight'})
    plt.show()                
    
    return df



#%% Call Plotter

if __name__ == '__main__':
    
    # xscale = {'value': 'log', 'base': 2}
    # yscale = {'value': 'log', 'base': 2}
    df = plot_example_run_averages(
        execset=10,
        save_fig=True,
        # xscale=xscale,
        # yscale=yscale
        )
    
