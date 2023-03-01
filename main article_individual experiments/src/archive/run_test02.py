# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 23:42:13 2021

@author: John Meluso
"""

import itertools as it
import matplotlib.pyplot as plt
import model_team as mt
import numpy as np
import pandas as pd
import seaborn as sns


if __name__ == "__main__":
    
    #%% Run simulation
    
    # Create all parameters
    index_degree = 1
    params = {
        'agent_fn': ['edgewt-root','edgewt-sphere'],
        'net_size': [10],
        'net_type': ['small-world', 'power']
        }
    runs = 100
    steps = 25
    metrics = ['team_degrees','avg_uniform','avg_weighted']
    
    # Create run spec and result dictionaries
    runspec2results = {
        'id':       [],
        'agent_fn': [],
        'net_size': [],
        'net_type': [],
        'run_num':  [],
        'team_degrees':     [],
        'avg_uniform':  [],
        'avg_weighted': []
        }
    net2fn2metric2results = {}
    for agent_fn in params['agent_fn']:
        net2fn2metric2results[agent_fn] = {}
        for net in params['net_type']:
            net2fn2metric2results[agent_fn][net] = {}
            for metric in metrics:
                net2fn2metric2results[agent_fn][net][metric] = []

    # Iterate through params
    for ii, (agent_fn, net_size, net_type) \
        in list(enumerate(it.product(*params.values()))):
        for run in range(runs):
        
            # Create team
            team = mt.Team(net_type, net_size, agent_fn)
            
            # Step team forward
            for step in range(steps):
                team.step()
                
            # Append to run specs dictionary
            runspec2results['id'].append(str(ii) + '_' + str(run))
            runspec2results['agent_fn'].append(agent_fn)
            runspec2results['net_size'].append(net_size)
            runspec2results['net_type'].append(net_type)
            runspec2results['run_num'].append(run)
            runspec2results['team_degrees'].append(
                np.array(team.degree)[:,index_degree])
            runspec2results['avg_uniform'].append(np.average(np.array(
                    team.history.copy()),
                    axis=1))
            runspec2results['avg_weighted'].append(np.average(np.array(
                    team.history.copy()),
                    weights=np.array(team.degree)[:,index_degree],
                    axis=1))
            
            # Append to net2fn2metric2results dictionary
            net2fn2metric2results[agent_fn][net_type]['team_degrees'].append(
                np.array(team.degree)[:,index_degree])
            net2fn2metric2results[agent_fn][net_type]['avg_uniform'].append(
                np.average(np.array(team.history.copy()),axis=1))
            net2fn2metric2results[agent_fn][net_type]['avg_weighted'].append(
                np.average(
                    np.array(team.history.copy()),
                    weights=np.array(team.degree)[:,index_degree],
                    axis=1)
                )
        
    # Convert results lists to panda data frame and numpy arrays
    df = pd.DataFrame(runspec2results)
    for agent_fn in params['agent_fn']:
        for net in params['net_type']:
            for metric in metrics:
                net2fn2metric2results[agent_fn][net][metric] = np.array(
                    net2fn2metric2results[agent_fn][net][metric])
                net2fn2metric2results[agent_fn][net]['avg_' + metric] \
                    = np.average(np.array(
                    net2fn2metric2results[agent_fn][net][metric]), axis=0)

                    
    #%% Plot results
                    
    # Create subplots
    fig = plt.figure(figsize=(20,14))
    axs = fig.subplots(len(params['agent_fn']), len(params['net_type']))
    for ii, agent_fn in enumerate(params['agent_fn']):
        for jj, net_type in enumerate(params['net_type']):
            axs[ii,jj].plot(
                net2fn2metric2results[agent_fn][net_type]['avg_avg_uniform']
                )
            axs[ii,jj].plot(
                net2fn2metric2results[agent_fn][net_type]['avg_avg_weighted']
                )
            axs[ii,jj].set_title(agent_fn + "_" + net_type)
    plt.show()
    