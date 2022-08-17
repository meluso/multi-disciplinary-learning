# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:43:18 2022

@author: jam
"""

# Import libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Import source files
import util.data as ud

All = slice(None)

def table_graph_metrics():
    
    # Load data
    execset = 7
    loc = f'../data/sets/execset{execset:03}_stats_by_graph.pickle'
    df = ud.load_pickle(loc)

    # Drop column levels
    df = df.loc[:,(All,'mean')]
    df.columns = df.columns.droplevel(1)
    mask = [col for col in df.columns if col != 'team_performance' and
            col != 'team_productivity']
    df = df.loc[:,mask]

    # Group by team size and team graph since graphs are the same
    df = df.groupby(['team_size','team_graph']).mean()
    
    # Subset by team size
    df = df.xs(9, level='team_size').reset_index()
    
    # Create latex table
    
    
    

if __name__ == '__main__':
    table_graph_metrics()