# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:38:02 2022

@author: John Meluso
"""

# import source files
import calc_stats_all as csa
import calc_stats_by_graph as csbg
import calc_stats_vs_baseline as csvb
import data_slice_functions as dsf


if __name__ == '__main__':
    
    # Select execution set
    execset = 1
    model = '3xx'
    base_graph = 'empty'
    
    #%% Dask-based reduction functions
    nn=6 # Number of workers
    dsf.slice_teamfn_is_nbhdfn(execset, model, num_workers=nn) # Slice data by function and model
    csa.describe_execset(execset, num_workers=nn) # Calculate stats for full execset by graph and function
    csbg.describe_execset(execset, num_workers=nn) # Calculate outcome stats by graph alone
    
    #%% Pandas-only reduction files
    csvb.merge_baseline_execset(execset, base_graph) # Baseline stats vs empty graph
