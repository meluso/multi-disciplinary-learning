# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:28:36 2022

@author: John Meluso
"""

#%% Import libraries

# Import libraries
from itertools import product
import numpy as np
from numpy import log10
import pandas as pd
from patsy import dmatrices
from patsy import ModelDesc
from patsy import Term
from patsy import EvalFactor as EvalFactor
from patsy import LookupFactor as LookupFactor
from sklearn.ensemble import RandomForestRegressor

# Import src files
from classes.Time import Time
import util.analysis as ua


#%% Regression utilities

def normalize_df(df, out_var, out_log, connected=True):
    
    norm_cols = [col for col in ua.get_graph_vars(connected)]
    if out_log != 'linear': norm_cols.append(out_var)
    
    # Cycle through columns in dataframe to see if they need normalizing
    for col in df.columns:
        if col in norm_cols:
            
            # Don't normalize if already normalized
            min_is_zero = abs(df[col].min() - 0) < 0.01
            max_is_one = abs(df[col].max() - 1) < 0.01
            if not(min_is_zero) or not(max_is_one):
                df[col] = normalize_col(df[col])
            
    return df


def normalize_col(df_col):
    return (df_col - df_col.mean()) / (df_col.max() - df_col.min())


#%% Variable builders

def get_fn_vars(controls='both'):
    
    if controls == 'metrics':
        fn_vars = [
            'team_fn_diff_integral',
            'team_fn_diff_peaks',
            'team_fn_alignment',
            'team_fn_interdep'
            ]
    elif controls == 'fes':
        fn_vars = ['C(team_fn)']
    elif controls == 'both':
        fn_vars = [
            'C(team_fn)',
            'team_fn_diff_integral',
            'team_fn_diff_peaks',
            'team_fn_alignment',
            'team_fn_interdep'
            ]
    else:
        raise RuntimeError(f'Function controls input {controls} is not valid.')
        
    return fn_vars


#%% Equation builders

def build_equations():
    
    # Connected & function difficulty configurations
    connected = [True, False]
    difficulties = ['both']
    
    # Iteratively build equations
    equations = {}
    props = {}
    for index, (conn, diff) in enumerate(product(connected, difficulties)):

        # Construct base equation
        equation = ModelDesc([Lookup('team_performance')],
                             [Lookup('run_step'),
                              Lookup('team_size'),
                              Lookup('agent_steplim')])

        # Add network and function variables
        equation = add_vars(equation, ua.get_graph_vars(conn))
        equation = add_vars(equation, get_fn_vars(diff))

        # Save equation to list of equations
        equations[index] = equation

        # Save equation properties
        props[index] = {
            'connected': conn,
            'difficulties': diff
            }
    
    return equations, props

def Lookup(var): return Term([LookupFactor(var)])

def Eval(var): return Term([EvalFactor(var)])

def add_vars(desc, extra_vars):
    desc.rhs_termlist += [Eval(p) for p in extra_vars]
    return desc


#%% Regression models

def run_random_forest(time, df, equation):
    
    # Get data for regression
    y, X = dmatrices(equation, data=df, return_type='matrix')
    y = y.ravel()
    cols = X.design_info.column_names
    time.update('Design matrices built')
    
    # Fit random forest
    forest = RandomForestRegressor(n_jobs=-1)
    forest = forest.fit(X, y)
    
    # Extract feature importances
    importances = forest.feature_importances_
    stdev = np.std(
        [tree.feature_importances_ for tree in forest.estimators_],
        axis=0
        )
    time.update('Random forest results generated')
    
    return time, cols, importances, stdev


#%% Running code

def run_all(execset, model):
    
    # Create timer
    time = Time()
    time.begin('all', 'random forest', 'models')
    
    # Load data
    loc = '../data/sets/'
    filename = f'execset{execset:03}_model{model}_team_is_nbhd.pickle'
    df = pd.read_pickle(loc + filename)
    time.update('Dataframe loaded')
            
    # Normalize data
    df = normalize_df(df, 'team_performance', 'linear', True)
    time.update('Dataframe normalized')
    
    # Construct equations
    equations, props = build_equations()
    time.update('Equations built')
    
    # Create results list
    results = {}
    
    # Iterate over equations
    for index, equation in equations.items():
        
        # Run regressions
        time, cols, imps, stdev = run_random_forest(time, df, equation)
        time.update(f'Random forest {index+1} of {len(equations)} complete')
        
        # Save result
        results[index] = {
            'variables': cols,
            'importances': imps,
            'stdevs': stdev,
            }
        
    # Save results
    filename = '../data/analysis/rf_features.pickle'
    pd.to_pickle(results, filename)
    time.end('Random forest','models')
    
    return results


if __name__ == '__main__':
    
    results = run_all(execset=1, model='3xx')