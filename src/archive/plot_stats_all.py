# -*- coding: utf-8 -*-
'''
Created on Fri Nov 19 14:54:48 2021

@author: John Meluso
'''

# Import libraries
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Import model files
from classes.Time import Time
import util.analysis as ua
import util.data as ud  
import util.plots as up


def plotter(time, file_in, file_out, activity, model, subset_perf, plot_opts,
            test_mode = False):
    '''Plots run means from file_in to file_out.'''
    
    # Mark Time
    time.begin('Model', activity, model)
    
    # Load run data
    params2stats = ud.load_pickle(file_in)
    All = slice(None)
    
    # Create pdf document
    pdf = PdfPages(file_out)
    
    # Mark Time
    time.update('Data loaded')
    
    # Get level indices
    params = up.level_indices(params2stats)
    
    # Create base dictionary
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
    
    lvl = tuple(model2_level2key.keys())
    
    # Create empty slices array, levels array, and dict options array
    key_array = []
    dct_array = []
    
    # Loop through functions
        
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
            
    # Mark Time
    time.update('Indeces built')
        
    # Subset down to just performance
    params2stats = params2stats['team_performance']
        
    # Mark Time
    time.update('Built params2stats')
        
    # Get number of figures
    nfigs = len(key_array)
    
    # Set theme for all plots
    sns.set_theme()
    sns.set(rc={'figure.figsize':(10,7.5)})
            
    for ii, (key, dct) in enumerate(zip(key_array, dct_array)):
        
        # Test mode
        if test_mode:
            if ii > 3: break
        
        # Get function subset of data
        data = params2stats.loc[key,:].reset_index()
        
        # Get label names
        model = dct['model']
        tmfn = dct['team_fn']
        gpnm = dct['group_name']
        gpfn = dct['group_fn']
        
        # Create one page for each function
        g = sns.relplot(
            kind='line',
            x='run_step',
            y=plot_opts['y_var'],
            hue='team_graph_type',
            row='agent_steplim',
            col='team_size',
            ci=None,
            facet_kws=dict(margin_titles=True),
            data=data
            )
        g.set_axis_labels('Time Step',plot_opts['y_label'])
        g.set_titles(
            col_template='Max. Possible Step: {col_name}',
            row_template='Team Size: {row_name}'
            )
        title = f'Model: {model} | Team Function: {tmfn} | ' \
            + f'{gpnm}: {gpfn}'
        g.figure.suptitle(title)
        g.figure.subplots_adjust(top=0.9)

        # Then, go to a new page
        pdf.savefig(g.figure)
        
        # Mark Time
        time.update(f'Page {ii + 1:03}/{nfigs} complete')
    
    # Close the pdf
    pdf.close()
    
    # Mark Time
    time.end(activity, model)
    

def plot_means_all(time, execset, models):
    
    for model in models:
    
        # Set filenames
        file_in = f'../data/sets/execset{execset:03}_stats.pickle'
        file_out = f'../figures/exploration/plots_means_all_{model}.pdf'
        activity = 'Plot Means'
        subset_perf = True
        plot_opts = {
            'y_var': 'mean',
            'y_label': 'Mean Team Performance'
            }
        
        # Run plotter
        plotter(time, file_in, file_out, activity, model, subset_perf, plot_opts)

def plot_vs_empty(time, execset, models):
    
    for model in models:
    
        # Set plotter inputs
        file_in = f'../data/sets/execset{execset:03}_vs_empty.pickle'
        file_out = f'../figures/exploration/plots_means_vs_empty_{model}.pdf'
        activity = 'Plot Vs Empty'
        subset_perf = True
        plot_opts = {
            'y_var': 'diff_mean',
            'y_label': 'Diff. Perf. Means: Shown - Empty'
            }
        
        # Run plotter
        plotter(time, file_in, file_out, activity, model, subset_perf, plot_opts)
    
    
if __name__ == '__main__':
    
    # Initialize time array
    time = Time()
    
    # Run parameters
    plot = 'vs_empty'
    execset = 7
    # models = ua.get_execset_models(execset)
    models = ['2x']
    
    if plot == 'stats' or plot=='both':
        plot_means_all(time, execset, models)
    if plot == 'vs_empty' or plot=='both':
        plot_vs_empty(time, execset, models)
    
        
    
    
    
    
    
    