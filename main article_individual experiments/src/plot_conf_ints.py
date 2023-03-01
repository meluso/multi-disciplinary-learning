# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:53:58 2022

@author: jam
"""

# Import libraries
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Import model files
from classes.Time import Time
import util.data as ud
import util.plots as up
        

def plotter(time, file_in, file_out, name, model):
    '''Plots run means from file_in to file_out.'''
    
    # Mark Time
    time.begin('model', name, model)
    
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
    
    # Set theme for all plots
    plt.style.use('seaborn')
            
    for ii, (idx, dct) in enumerate(zip(key_array, dct_array)):
        
        # Get function subset of data
        data = params2stats.loc[idx]
        
        # Plot the run data by graph, team size
        figure = build_ci_plot(data, dct)

        # Then, go to a new page
        pdf.savefig(figure)
        
        # Mark Time
        time.update(f'Page {ii + 1:03}/{nfigs} complete')
    
    # Close the pdf
    pdf.close()
    
    # Mark Time
    time.end(name, model)
    
def build_indices(model, params):
    
    # Create empty slices array and dict options array
    key_array = []
    dct_array = []
    All = slice(None)
    
    # Get Model level2key dictionary
    model2_level2key = dict(
        team_size = All,
        team_graph_type = All,
            team_graph_k = ['na',str(2)],
            team_graph_m = ['na',str(2)],
            team_graph_p = ['na',str(0.5)],
        team_fn_type = All,
            team_fn_exponent = ['na','degree'],
            team_fn_frequency = ['na','uniform'],
            team_fn_weight = ['na','node'],
        agent_fn_type = All,
            agent_fn_exponent = ['na','degree'],
            agent_fn_frequency = ['na','uniform'], 
            agent_fn_weight = ['na','node'],
        agent_steplim = All,
        run_step = All,
        )
    model3_level2key = dict(
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
            agent_fn_exponent = ['na','degree'],
            agent_fn_frequency = ['na','uniform'], 
            agent_fn_weight = ['na','node'],
        agent_steplim = All,
        run_step = All,
        )
    
    # Loop through functions by model type
        
    if model == '2x':
        
        # Delete previously-removed entries
        level2key = model2_level2key.copy()
        del level2key['agent_fn_type']
        del level2key['agent_fn_exponent']
        del level2key['agent_fn_frequency']
        del level2key['agent_fn_weight']
    
        for tmfn in params['team_fn_type']['values']:
            
            # Place team function
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
        
        # Copy dictionary
        level2key = model2_level2key.copy()
    
        for tmfn in params['team_fn_type']['values']:    
            for agfn in params['agent_fn_type']['values']:
                
                # Place team function
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
                
    elif model == '3xx':
        
        # Delete previously-removed entries
        level2key = model3_level2key.copy()
        del level2key['agent_fn_type']
        del level2key['agent_fn_exponent']
        del level2key['agent_fn_frequency']
        del level2key['agent_fn_weight']
        
        for tmfn in params['team_fn_type']['values']:
            for nhfn in [nhfn for nhfn in params['nbhd_fn_type']['values'] \
                         if (nhfn != 'na')
                         and (tmfn == nhfn)]:
                
                # Place team function
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
                
    elif model == '3fx':
        
        # Copy dictionary
        level2key = model3_level2key.copy()
        
        for tmfn in params['team_fn_type']['values']:
            for nhfn in [nhfn for nhfn in params['nbhd_fn_type']['values'] \
                         if (nhfn != 'na')
                         and (tmfn == nhfn)]:
                
                # Place team function
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
                
    elif model == '3xg':
        
        # Delete previously-removed entries
        level2key = model3_level2key.copy()
        del level2key['agent_fn_type']
        del level2key['agent_fn_exponent']
        del level2key['agent_fn_frequency']
        del level2key['agent_fn_weight']
        
        for tmfn in params['team_fn_type']['values']:
            for nhfn in [nhfn for nhfn in params['nbhd_fn_type']['values'] \
                         if (nhfn != 'na')
                         and (tmfn == nhfn)]:
                
                # Place team function
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
                
    elif model == '3fg':
        
        # Copy dictionary
        level2key = model3_level2key.copy()
        
        for tmfn in params['team_fn_type']['values']:
            for nhfn in [nhfn for nhfn in params['nbhd_fn_type']['values'] \
                         if (nhfn != 'na')
                         and (tmfn == nhfn)]:
                
                # Place team function
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

def build_ci_plot(df, dct):
        
    # Get label names
    model = dct['model']
    tmfn = dct['team_fn']
    gpnm = dct['group_name']
    gpfn = dct['group_fn']
    
    # Get lists of unique graphs and team sizes to slice by
    graphs = df.reset_index()['team_graph_type'].unique()
    sizes = df.reset_index()['team_size'].unique()
    
    # Create figure with 1 x len(sizes) subfigures
    fig, axs \
        = plt.subplots(1, len(sizes), sharey=True)
    
    # Iterate through sizes & graphs
    for ii, sz in enumerate(sizes):
        handles, labels = [], []
        for jj, gr in enumerate(graphs):
            
            # Slice data by size & graph
            data = df.xs(key=(gr, sz),level=('team_graph_type', 'team_size'))
            data = data.reset_index()
            
            # Plot confidence interval and line itself
            axs[ii].fill_between(
                x='run_step',
                y1='diff_ci_lo',
                y2='diff_ci_hi',
                alpha=.5,
                linewidth=0,
                data=data,
                label=str(gr)
                )
            axs[ii].plot(
                'run_step',
                'diff_mean',
                data=data,
                label=str(gr)
                )
            
            # Set title
            axs[ii].set_title(f'{sz} Agents')
            
    # Add SupTitle
    title = f'Model: {model} | Team Function: {tmfn} | {gpnm}: {gpfn}'
    fig.suptitle(title, fontsize=16)
    fig.supxlabel('Time Step',y=0.12)
    fig.supylabel('Diff. Perf. Means: Shown - Empty')
    
    # Add legend
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc=8, ncol=4)
    
    # Close figure
    fig.tight_layout(rect=(0,0.1,1,1))
    plt.close(fig)
    
    return fig

def plot_conf_ints(models):
    
    # Initialize time array
    time = Time()
    
    for model in models:
        file_in = f'../data/sets/model{model}_vs_empty.pickle'
        file_out = f'../figures/conf_ints/plots_conf_ints_{model}.pdf'
        name = 'Plot Confidence Intervals Vs Empty Graph'
        plotter(time, file_in, file_out, name, model)
    
    
if __name__ == '__main__':
    
    # Run parameters
    models = [
        '2x',
        '3xx',
        '3xg'
        ]
    
    # Create PDFs for the execpairs
    plot_conf_ints(models)
        
    
    