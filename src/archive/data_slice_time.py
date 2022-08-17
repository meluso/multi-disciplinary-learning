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


def slice_data(model, slices, slice_name, merge_vars=True, repartition=None):
    
    # Start timer
    time = Time()
    time.begin('Model', slice_name, f'{model}')
    
    # Get execset and exec_list
    execset, exec_list = ua.get_model_execsets(model)
    
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
    
    # Build variables to drop
    drop_vars = [
        'model_type', 'run_ind',
        'agent_fn_type', 'agent_fn_difficulty', 'agent_steplim',
        'team_graph_density',
        'team_graph_k', 'team_graph_m', 'team_graph_p',
        'agent_fn_weight', 'agent_fn_frequency', 'agent_fn_exponent'
    ]
    
    # Set smaller datatypes
    points = points.astype(dtype=ua.get_dtypes())
    
    # Slice by index
    for key, value in slices.items():
        points = points[points[key] == value]
        drop_vars.append(key)
        if (repartition is not None) and (repartition[key] is not None):
            points = points.repartition(
                npartitions=points.npartitions // repartition[key]
                )

    # Drop variables & convert back to pandas dataframe
    points = points.drop(columns=drop_vars).compute()
    
    # Return stat time
    time.end(slice_name, f'{model}')

    # Pickle stats
    if paramset == 'test': paramset = 0
    filename = f'../data/sets/model{model}_{slice_name}.pickle'
    points.to_pickle(filename)
    
def slicer(model, slices, slice_name, repartition=None):
    fn = lambda: slice_data(model, slices, slice_name, repartition)
    points = ua.run_dask_cluster(fn)
    return points

def slice_time05(model):
    slices = {'run_step': 5}
    slice_name = 'time05'
    return slicer(model, slices, slice_name)

def slice_time10(model):
    slices = {'run_step': 10}
    slice_name = 'time10'
    return slicer(model, slices, slice_name)

def slice_time05_step10(model):
    slices = {
        'run_step': 5,
        'agent_steplim': 1.0
        }
    slice_name = 'time05_step10'
    repartition = {
        'run_step': 26,
        'agent_steplim': None
        }
    return slicer(model, slices, slice_name, repartition)

def slice_time10_step10(model):
    slices = {
        'run_step': 10,
        'agent_steplim': 1.0
        }
    slice_name = 'time10_step10'
    repartition = {
        'run_step': 26,
        'agent_steplim': None
        }
    return slicer(model, slices, slice_name, repartition)

if __name__ == '__main__':
    
    points = slice_time05('3xx')
    points = slice_time05('3xg')
    points = slice_time10('3xx')
    points = slice_time10('3xg')