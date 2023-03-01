# -*- coding: utf-8 -*-
'''
Created on Fri Nov 19 14:54:48 2021

@author: John Meluso
'''

# Import libraries
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd

# Import model files
from classes.Time import Time
import util.analysis as ua
import util.data as ud  
import util.plots as up
import util.variables as uv


colors=[
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a',
    '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
    '#17becf', '#9edae5'
    ]

def plotter(time, file_in, file_out, activity, model, plot_opts,
            test_mode = False):
    '''Plots run means from file_in to file_out.'''
    
    # Mark Time
    time.begin('Model', activity, model)
    
    # Load run data
    params2stats = ud.load_pickle(file_in)
    
    # Create pdf document
    pdf = PdfPages(file_out)
    
    # Mark Time
    time.update('Data loaded')
    
    # Get level indices
    params = up.level_indices(params2stats)
    
    # Build indices
    key_array, dct_array = build_indices(model, params)
            
    # Mark Time
    time.update('Indeces built')
        
    # Get number of figures
    nfigs = len(key_array)
    
    # Subset down to just performance
    cols = (list(uv.get_outcomes().keys()),['mean'])
    
    # Set theme for all plots
    plt.style.use('seaborn')
            
    for ii, (key, dct) in enumerate(zip(key_array, dct_array)):
        
        # Test mode
        if test_mode:
            if ii > 3: break
        
        # Get function subset of data
        data = params2stats.loc[key,cols].droplevel(level=1,axis=1)
        data = pd.melt(
            data.reset_index(),
            id_vars=data.index.names,
            var_name='outcome'
            )
        
        # Plot the run data by graph, team size
        figure = build_outcomes_plot(data, dct)

        # Then, go to a new page
        pdf.savefig(figure)
        
        # Mark Time
        time.update(f'Page {ii + 1:03}/{nfigs} complete')
    
    # Close the pdf
    pdf.close()
    
    # Mark Time
    time.end(activity, model)
    
def build_indices(model, params):
    
    # Create empty slices array and dict options array
    key_array = []
    dct_array = []
    All = slice(None)
    
    # Get Model level2key dictionary
    model2_level2key = dict(
        model_type = model,
        team_size = All,
        team_graph_type = All,
            team_graph_k = ['na',str(2)],
            team_graph_m = ['na',str(2)],
            team_graph_p = ['na',str(0.5)],
        team_fn_type = All,
            team_fn_exponent = ['na','degree'],
            team_fn_frequency = ['na','uniform'],
            team_fn_weight = ['na','node'],
        nbhd_fn_type = All,
            nbhd_fn_exponent = All,
            nbhd_fn_frequency = All,
            nbhd_fn_weight = All,
        agent_fn_type = All,
            agent_fn_exponent = ['na','degree'],
            agent_fn_frequency = ['na','uniform'], 
            agent_fn_weight = ['na','node'],
        agent_steplim = All,
        run_step = All,
        )
    
    model3_level2key = dict(
        model_type = model,
        team_size = All,
        team_graph_type = All,
            team_graph_k = ['na',str(2)],
            team_graph_m = ['na',str(2)],
            team_graph_p = ['na',str(0.5)],
        team_fn_type = All,
            team_fn_exponent = ['na','degree'],
            team_fn_frequency = ['na','uniform'],
            team_fn_weight = ['na','node'],
        nbhd_fn_type = All,
            nbhd_fn_exponent = ['na','degree'],
            nbhd_fn_frequency = ['na','uniform'],
            nbhd_fn_weight = ['na','node'],
        agent_fn_type = All,
            agent_fn_exponent = All,
            agent_fn_frequency = All, 
            agent_fn_weight = All,
        agent_steplim = All,
        run_step = All,
        )
    
    # Loop through functions by model type
        
    if model == '2x':
    
        for tmfn in params['team_fn_type']['values']:
            
            # Place team function
            level2key = model2_level2key
            level2key['team_fn_type'] = tmfn
            
            # Append values to arrays
            key_array.append(tuple(level2key.values()))
            
            # Get plotting options
            dct_array.append({
                'model': model,
                'team_fn': tmfn,
                'group_name': 'Agent Function',
                'group_fn': 'Not Applicable'
                })
                
    elif model == '2f':
    
        for tmfn in params['team_fn_type']['values']:    
            for agfn in params['agent_fn_type']['values']:
                
                # Place team function
                level2key = model2_level2key
                level2key['team_fn_type'] = tmfn
                level2key['agent_fn_type'] = agfn
                
                # Append values to arrays
                key_array.append(tuple(level2key.values()))
                
                # Get plotting options
                dct_array.append({
                    'model': model,
                    'team_fn': tmfn,
                    'group_name': 'Agent Function',
                    'group_fn': agfn
                    })
                
    elif model == '3xx' or model == '3fx' or model == '3xg' or  model == '3fg':
        
        for tmfn in params['team_fn_type']['values']:
            for nhfn in [nhfn for nhfn in params['nbhd_fn_type']['values'] \
                         if (nhfn != 'na')]:
            
                # Place team function
                level2key = model3_level2key
                level2key['team_fn_type'] = tmfn
                level2key['nbhd_fn_type'] = nhfn
                
                # Append values to arrays
                key_array.append(tuple(level2key.values()))
                
                # Get plotting options
                dct_array.append({
                    'model': model,
                    'team_fn': tmfn,
                    'group_name': 'Neighborhood Function',
                    'group_fn': nhfn
                    })
                
    else:
        
        raise RuntimeError
        
    return key_array, dct_array
	
def build_outcomes_plot(df, dct):
        
    # Get label names
    model = dct['model']
    tmfn = dct['team_fn']
    gpnm = dct['group_name']
    gpfn = dct['group_fn']
    
    # Get lists of unique outcomes and team graphs to slice by
    df = df.reset_index()
    outcomes = df['outcome'].unique()
    sizes = df['team_size'].unique()
    graphs = df['team_graph_type'].unique()
    
    # Create figure with 1 x len(sizes) subfigures
    fig, axs = plt.subplots(len(outcomes), len(sizes), sharey='row',
                            figsize=(15,12)
                            )
    
    # Iterate through sizes & graphs
    for ii, oc in enumerate(outcomes):
        axs[ii,0].set_ylabel(oc.capitalize().replace('_',' '))
        for jj, sz in enumerate(sizes):
            axs[0,jj].set_title(f'{sz} Agents')
            for kk, gr in enumerate(graphs):
            
                handles, labels = [], []
                
                # Slice data by size & graph
                mask = (df['team_size'] == sz) & (df['outcome'] == oc) \
                    & (df['team_graph_type'] == gr)
                data = df[mask]
        
                # Plot line
                axs[ii,jj].plot('run_step', 'value', data=data, label=gr,
                                color=colors[kk])
            
    # Add SupTitle
    title = f'Model: {model} | Team Function: {tmfn} | {gpnm}: {gpfn}'
    fig.suptitle(title, fontsize=16)
    fig.supxlabel('Time Step', y=0.12)
    
    # Add legend
    handles, labels = axs[-1,-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc=8, ncol=4)
    
    # Close figure
    fig.tight_layout(rect=(0,0.1,1,1))
    plt.close(fig)
    
    return fig    

def plot_means_all(time, execset, models):
    
    for model in models:
    
        # Set filenames
        file_in = f'../data/sets/execset{execset:03}_stats.pickle'
        file_out = f'../figures/outcomes/outcome_means_all_{model}.pdf'
        activity = 'Plot Means'
        test_mode = False
        plot_opts = {
            'y_var': 'outcome',
            'y_label': 'Mean of Displayed Team Outcome'
            }
        
        # Run plotter
        plotter(time, file_in, file_out, activity, model, plot_opts, test_mode)
    
    
if __name__ == '__main__':
    
    # Initialize time array
    time = Time()
    
    # Run parameters
    execset = 7
    models = ua.get_execset_models(execset)
    # models = ['2x']
    
    plot_means_all(time, execset, models)
    
        
    
    
    
    
    
    