# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:05:44 2022

@author: John Meluso
"""

#%% Import libraries

# Import libraries
from itertools import product
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.iolib.summary2 as su2
import statsmodels.iolib.smpickle as pickle

# Import src files
from classes.Time import Time
import util.data as ud


#%% Regression Functions

def regress_points(model, mode, out_var, out_log):
    '''Run regression on points.'''

    # Start timer
    time = Time()
    time.begin('Model', f'{mode} regression on {out_var}', model)

    # Load data
    filename = f'../data/sets/model{model}.pickle'
    df = ud.load_pickle(filename)
    df = normalize_df(df, out_var)
    time.update('Data loaded')

    # Construct variable equations and results list
    equations = select_formula(mode, out_var, out_log)
    results = []
    time.update('Equations constructed')

    # Iterate through equations
    for ii, eq in enumerate(equations):

        # Run dataframe through patsy
        y, X = dmatrices(eq, data=df, return_type='dataframe')
        time.update('Design matrices constructed')

        # Create OLS model
        mod = sm.OLS(y, X)
        res = mod.fit()

        # Add result to list
        results.append(res)
        time.update(f'Regression Model {ii+1} of {len(equations)}')

    time.end(f'{mode} regression on {out_var}', model)
    return results

def select_formula(mode, out_var, out_log=None):
    '''Constructs a regression formula from the columns of the dataframe
    according to the specified mode.'''
    
    # Always include team size
    if out_var == 'team_productivity':
        include_always = ['np.log10(run_step)']
    else:
        include_always = ['np.log10(run_step)','team_size']
    
    # Then select other include always variables based on mode
    if mode == 'fixed_by_fn_subtype':
        include_always.append('team_fn')
        include_always.append('nbhd_fn')
    elif mode == 'fixed_by_fn_type':
        include_always.append('team_fn_type') 
        include_always.append('nbhd_fn_type')
    elif mode == 'metrics':
        app = ['team_fn_alignment','team_fn_interdep','team_fn_difficulty']
        for aa in app: include_always.append(aa)
    
    # Add iterator variables
    include_iter = get_graph_vars()
    
    # Build equations
    return build_equation(include_always, include_iter, out_var, out_log)


def build_equation(include_always, include_iter, out_var, out_log=None):
    
    # Build array of equations
    eqs = []
    
    # Construct base equation
    if out_log is not None:
        eq_base = f'np.log{out_log}({out_var})' + ' ~ '
    else:
        eq_base = out_var + ' ~ '
    for ii, var in enumerate(include_always):
        if ii > 0:
            eq_base += ' + ' + var
        else:
            eq_base += var
    
    # Iteratively build single-variable models from base equation
    for var in include_iter:
        eqs.append(eq_base + ' + ' + var)
        
    # Build all-variable model from base equation
    all_vars = eq_base
    for var in include_iter:
        all_vars += ' + ' + var
    eqs.append(all_vars)
    
    return eqs


#%% Regression utilities

def normalize_df(df, out_var):
    
    list_of_lists = [get_graph_vars(), get_fn_vars('metrics'), [out_var]]
    norm_cols = [col for sub_list in list_of_lists for col in sub_list]
    
    for col in df.columns:
        if col in norm_cols:
            df[col] = normalize(df[col])
            
    return df


def normalize(df_col):
    return (df_col - df_col.mean()) / (df_col.max() - df_col.min())


def generate_table(model, slice_name, mode, results, out_var, out_log):
    
    # Get model names
    names = [f'{ii+1}' for ii, res in enumerate(results)]
    
    # Build variable order for table
    base_vars = ['Intercept','np.log10(run_step)','team_size']
    graph_vars = get_graph_vars()
    var_list = [base_vars, graph_vars]
    order = [var for sub_list in var_list for var in sub_list]

    # Construct table from results
    table = su2.summary_col(
        results,
        model_names=names,
        stars=True,
        regressor_order=order
        )
    
    # Set out_log variable
    if out_log is None:
        out_log = 'linear'
    else:
        out_log = f'log{out_log}'
    
    # Save Table to Pickle
    file = f'../data/regression/reg_{out_log}_{out_var}_{mode}_model{model}.pickle'
    pickle.save_pickle(table, file)
    
    # Save Table to Text File
    file = f'../figures/regression/reg_{out_log}_{out_var}_{mode}_model{model}.txt'
    with open(file, 'w') as output:
        output.write(table.as_text())
        
        
#%% Variable builders

def get_graph_vars():
    
    return [
        'team_graph_centrality_degree_mean',
        'team_graph_centrality_degree_stdev',
        'team_graph_centrality_eigenvector_mean',
        'team_graph_centrality_eigenvector_stdev',
        'team_graph_centrality_betweenness_mean',
        'team_graph_centrality_betweenness_stdev',
        'team_graph_nearest_neighbor_degree_mean',
        'team_graph_nearest_neighbor_degree_stdev',
        'team_graph_clustering',
        'team_graph_assortativity',
        'team_graph_pathlength',
        'team_graph_diameter'
        ]

def get_fn_vars(mode):
    
    if mode == 'metrics':
        return [
            'team_fn_difficulty',
            'team_fn_alignment',
            'team_fn_interdep'
            ]
    else:
        raise RuntimeError(f'Function variable mode {mode} not valid.')
        


#%% Running code

def run_regression(model, mode, out_var, out_log=None):
    results = regress_points(model, mode, out_var, out_log)
    return generate_table(model, mode, results, out_var, out_log)


if __name__ == '__main__':
    
    models = ['3xx','3xg']
    reg_types = ['fixed_by_fn_subtype']
    out_vars = ['team_productivity']
    out_log = 10
    
    for model, reg, out in product(models, reg_types, out_vars):
        run_regression(model, reg, out, out_log)
