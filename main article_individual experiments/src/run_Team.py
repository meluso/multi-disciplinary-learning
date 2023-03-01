# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 15:54:50 2021

@author: John Meluso
"""

# Import libraries
import datetime as dt
import networkx as nx
from numpy.random import default_rng

# Import model classes
import classes.Team as Team


# Create the random number generator
# rng = default_rng(seed=42)
rng = default_rng()


def run_Team(points, params, messages=True, graph=False):
    '''Function which runs one instance of the team model for a given set of
    input parameters and save the resulting data.'''
    
    # Print input parameters
    if messages: print('Run parameters: ' + str(params))
    
    # Start timer
    t_start = dt.datetime.now()
    
    # Pull parameters to variables and delete
    model_type = params['model_type']
    del params['model_type']
    
    num_steps = params['num_steps']
    del params['num_steps']
    
    run_ind = params['run_ind']
    del params['run_ind']
    
    # Initialize team with parameters
    team = getattr(Team,'TeamType' + str(model_type))(**params)
    
    # Execute specified number of steps
    for step in range(num_steps + 1):
        if step > 0: team.step()
        points.update(params, team, model_type, run_ind, step)
        
    # Print run duration
    t_run = dt.datetime.now()
    if messages: print('Run duration: ' + str(t_run - t_start) + '\n')
    
    # Plot the team
    if graph: nx.draw(team)
    
    # Return points
    return points


