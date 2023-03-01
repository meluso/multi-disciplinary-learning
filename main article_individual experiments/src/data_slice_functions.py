# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 17:21:38 2022

@author: John Meluso
"""

# Import libraries
import dask.dataframe as dd

# Import src files
from classes.Time import Time
import util.analysis as ua
import util.variables as uv


def slice_data(execset, model, slice_name):
    
    # Start timer
    time = Time()
    time.begin('Model', slice_name, f'{model}')
    
    # Get execset and exec_list
    exec_list = ua.get_execset_model_execs(execset, model)
    
    # Get list of files
    file_list, paramset = ua.get_file_list(exec_list)
    
    # Make a unified points dataframe
    points = dd.read_csv(file_list, dtype=uv.get_dtypes(), blocksize=None)
    
    # Combine team function columns
    points['team_fn'] = points['team_fn_type'] + '_' \
        + points['team_fn_weight'] + '_' \
        + points['team_fn_frequency'] + '_' \
        + points['team_fn_exponent']
    
    # Combine neighborhood function columns
    points['nbhd_fn'] = points['nbhd_fn_type'] + '_' \
        + points['nbhd_fn_weight'] + '_' \
        + points['nbhd_fn_frequency'] + '_' \
        + points['nbhd_fn_exponent']
        
    # Mask data to get only cases where team function equal neighborhood fn.
    points = points[(points.team_fn == points.nbhd_fn)]
    
    # Build variables to drop
    drop_vars = [
        'model_type', 'run_ind',
        'nbhd_fn_type', 'nbhd_fn_diff_integral', 'nbhd_fn_diff_peaks',
        'agent_fn_type', 'agent_fn_diff_integral', 'agent_fn_diff_peaks',
        'team_graph_density',
        'team_graph_k', 'team_graph_m', 'team_graph_p',
        'nbhd_fn_weight', 'nbhd_fn_frequency', 'nbhd_fn_exponent',
        'agent_fn_weight', 'agent_fn_frequency', 'agent_fn_exponent'
    ]

    # Drop variables & convert back to pandas dataframe
    points = points.drop(columns=drop_vars)
    
    # Pickle slice
    if paramset == 'test': paramset = 0
    loc = '../data/sets/'
    filename = f'execset{execset:03}_model{model}_{slice_name}.pickle'
    output = points.compute()
    output.to_pickle(loc + filename)
    
    # Return stat time
    time.end(slice_name, f'{model}')
    
def slicer(execset, model, slice_name, **kwargs):
    fn = lambda: slice_data(execset, model, slice_name)
    ua.run_dask_cluster(fn, **kwargs)

def slice_teamfn_is_nbhdfn(execset, model, **kwargs):
    slicer(execset, model, 'team_is_nbhd', **kwargs)

# if __name__ == '__main__':
#     execset = 8
#     slice_teamfn_is_nbhdfn(execset, '3xx')
#     slice_teamfn_is_nbhdfn(execset, '3xg')