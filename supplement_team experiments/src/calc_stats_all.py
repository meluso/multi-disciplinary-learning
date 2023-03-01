# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:16:46 2021

@author: John Meluso
"""

# Import libraries
import dask.dataframe as dd

# Import model functions
from classes.Time import Time
import util.analysis as ua
import util.variables as uv


def calc_stats_execset(execset):
    '''Calculates descriptive statistics for the provided dataframe.'''
    
    # Start timer
    time = Time()
    time.begin('Execution Set', 'Describe Points', f'{execset}')
    
    # Get exec_list
    exec_list = ua.get_execset_execs(execset)
    
    # Get list of files
    file_list, paramset = ua.get_file_list(exec_list)
            
    # Make a unified points dataframe and drop run index
    points = dd.read_csv(file_list, blocksize=None, dtype=uv.get_dtypes())
    points = points.drop(columns=uv.not_level_indices())
    
    # Groupby index columns and calculate aggregations
    ops = ['count','mean','std','var']
    groupagg = points.groupby(uv.all_level_indices()).aggregate(ops)
    
    # Convert from dask dataframe to pandas dataframe
    params2stats = groupagg.compute()
    
    # Return stat time
    time.end('Describe Points', f'{execset}')

    # Pickle stats
    if paramset == 'test': paramset = 0
    params2stats.to_pickle(f'../data/sets/execset{execset:03}_stats.pickle')
    
    return params2stats

def calc_stats_model(model, test=False):
    '''Calculates descriptive statistics for the provided dataframe & model.'''
    
    # Start timer
    time = Time()
    time.begin('model', 'Describe Subset', f'{model}')
    
    # Get execset and exec_list
    execset, exec_list = ua.get_model_execsets(model)
    
    # Get list of files
    file_list, paramset = ua.get_file_list(exec_list)
                
    # Reduce list for test
    if test: file_list = file_list[0:5]
            
    # Make a unified points dataframe and drop run index
    points = dd.read_csv(file_list, blocksize=None, dtype=uv.get_dtypes())
    points = points.drop(columns=ua.get_drop_vars(model))
    
    # Groupby index columns and calculate aggregations
    ops = ['count','mean','std','var']
    groupagg = points.groupby(ua.get_group_vars(model)).aggregate(ops)
    
    # Convert from dask dataframe to pandas dataframe
    params2stats = groupagg.compute()
    
    # Reduce data to specified model
    params2stats = params2stats.loc[model,:]
    
    # Return stat time
    time.end('Describe Subset', f'{model}')

    # Pickle stats
    if paramset == 'test': paramset = 0
    if test: model = model + 'TEST'
    params2stats.to_pickle(f'../data/sets/model{model}_stats.pickle')
    
    return params2stats

def describe_execset(execset, num_workers=4):
    fn = lambda: calc_stats_execset(execset)
    params2stats = ua.run_dask_cluster(fn, num_workers)
    return params2stats
        
def describe_execset_005():
    fn = lambda: calc_stats_execset(5)
    params2stats = ua.run_dask_cluster(fn)
    return params2stats

def describe_execset_007():
    fn = lambda: calc_stats_execset(7)
    params2stats = ua.run_dask_cluster(fn)
    return params2stats

def describe_execset_008():
    fn = lambda: calc_stats_execset(8)
    params2stats = ua.run_dask_cluster(fn)
    return params2stats

def describe_model(model):
    fn = lambda: calc_stats_model(model)
    params2stats = ua.run_dask_cluster(fn)
    return params2stats
        
def describe_model_2x():
    fn = lambda: calc_stats_model('2x')
    params2stats = ua.run_dask_cluster(fn)
    return params2stats

def describe_model_3xx():
    fn = lambda: calc_stats_model('3xx')
    params2stats = ua.run_dask_cluster(fn)
    return params2stats

def describe_model_3xg():
    fn = lambda: calc_stats_model('3xg')
    params2stats = ua.run_dask_cluster(fn)
    return params2stats


# if __name__ == '__main__':
#     params2stats = describe_execset_008()
#     params2stats = describe_model_3xx()
#     params2stats = describe_model_3xg()
    