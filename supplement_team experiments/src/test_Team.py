# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:17:34 2021

@author: John Meluso
"""
#%% Load libraries

# Import libraries
import datetime as dt
import matplotlib.pyplot as plt
from numpy import mod
from numpy.random import default_rng
import seaborn as sns

# Import model classes
from classes.Points import Points

# Import model methods
import run_Team as rt
import util.data as ud
import util.params as up


def test_Team(run_mode='both', num_runs='all', messages=False, graph=False):
    
    #%% Run all combos
    
    # Start timer
    t_start = dt.datetime.now()
    
    # Pick random param in all_params
    if num_runs == 1:
        rng = default_rng()
        # rng = default_rng(seed=42)
        paramset = 'TestOnce'
        casenum = rng.integers(up.count_params(paramset))
        cases = [up.get_params(paramset,get_cases=[casenum])]
    else:
        rng = default_rng(seed=42)
        cases = up.get_params('TestModels',all_cases=True)
    
    # Param retrieval time
    t_param = dt.datetime.now()
    print('Parameter retrieval duration: ' + str(t_param - t_start) + '\n')
    
    # Create data object
    points = Points()
    
    ### Run team ###########################################################
    for ii, case in enumerate(cases):
        points = rt.run_Team(points, case, messages, graph)
        t_mid = dt.datetime.now() - t_start
        if mod(ii+1,100) == 0:
            print(f'Run {ii+1} of {len(cases)} complete after {t_mid}.')
    ########################################################################
        
    # All runs time
    t_runs = dt.datetime.now()
    print('All runs duration: ' + str(t_runs - t_start) + '\n')
    
    # Manage results
    points_df = ud.manage_results(run_mode, points, ud.get_test_loc())
    
    # Results management time
    t_results = dt.datetime.now()
    print('Data management duration: ' + str(t_results - t_runs) + '\n')
    print('Test duration: ' + str(t_results - t_start) + '\n')
    
    return cases, points_df

def plot_test(points, num_runs='all'):
    
    # Plot results
    sns.set_theme()
    
    if num_runs == 1:
        g = sns.relplot(
            x='run_step',
            y='team_performance',
            ci=None,
            kind='line',
            data=points)
        fig = plt.figure(g.figure)
        plt.savefig('../figures/tests/test_Team_one.png',dpi=300)
    else:
        g = sns.relplot(
            x='run_step',
            y='team_performance',
            col='model_type',
            row='team_fn_type',
            hue='team_graph_type',
            size='nbhd_fn_type',
            style='agent_fn_type',
            ci=None,
            kind='line',
            data=points)
        fig = plt.figure(g.figure)
        plt.savefig('../figures/tests/test_Team_all.png',dpi=300)
    

if __name__ == '__main__':
    
    # Run test
    cases, points = test_Team(num_runs=1, messages=True, graph=True)
    # cases, points = test_Team()
    
    # Plot results
    plot_test(points)
    