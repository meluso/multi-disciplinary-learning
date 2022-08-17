# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:35:33 2022

@author: John Meluso
"""

#%% Import libraries

# Import libraries
from itertools import product
from numpy import log10
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.iolib.summary2 as su2
import statsmodels.iolib.smpickle as smp

# Import src files
from classes.Time import Time
import util.data as ud

# Columns loaded from pickle:
# ['team_size', 'team_graph_type', 'team_fn_type', 'run_step', 'agent_steplim',
#  'team_graph_centrality_degree_mean',
#  'team_graph_centrality_degree_stdev',
#  'team_graph_centrality_eigenvector_mean',
#  'team_graph_centrality_eigenvector_stdev',
#  'team_graph_centrality_betweenness_mean',
#  'team_graph_centrality_betweenness_stdev',
#  'team_graph_nearest_neighbor_degree_mean',
#  'team_graph_nearest_neighbor_degree_stdev',
#  'team_graph_clustering',
#  'team_graph_assortativity',
#  'team_graph_pathlength',
#  'team_graph_diameter',
#  'team_fn_diff_integral',
#  'team_fn_alignment',
#  'team_fn_interdep',
#  'team_performance',
#  'team_productivity',
#  'team_fn_weight',
#  'team_fn_frequency',
#  'team_fn_exponent',
#  'team_fn',
#  'nbhd_fn']


#%% Regression utilities

def normalize_df(df, out_var, out_log, connected=True):
    
    norm_cols = [col for col in get_graph_vars(connected)]
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


def build_equation(out_var, out_log='linear', connected=True):
    
    # Construct base variables
    if out_var == 'team_productivity':
        base_vars = ['log10(run_step)', 'team_fn', 'team_size', 'team_size^2',
                     'C(agent_steplim)']
    else:
        base_vars = ['log10(run_step + 1)', 'team_fn', 'team_size',
                     'team_size^2', 'C(agent_steplim)']
    
    # Construct base equation
    if out_log != 'linear':
        equation = f'log{out_log}({out_var})' + ' ~ '
    else:
        equation = out_var + ' ~ '
    for ii, var in enumerate(base_vars):
        if ii > 0:
            equation += ' + ' + var
        else:
            equation += var
        
    # Add remaining graph variables
    for var in get_graph_vars(connected):
        equation += ' + ' + var
        
    return equation


def add_to_dict(results_dict, results, index, **kwds):
    
    # Add results to index
    results_dict[index] = dict(results = results)
    
    # Add other keywords to index
    for key, value in kwds.items():
        results_dict[index][key] = value

    return results_dict


def generate_table(results, connected=True):
    
    # Get model names
    names = []
    res_list = []
    for key, value in results.items():
        model = value['model']
        out_var = value['out_var']
        out_log = value['out_log']
        names.append(f'M{model} w/ ({out_log})\n{out_var}')
        res_list.append(value['results'])
    res_list = [res['results'] for res in results.values()]
    
    # Build variable order for table
    base_vars = ['Intercept','run_step','log10(run_step)','team_size']
    graph_vars = get_graph_vars(connected)
    var_list = [base_vars, graph_vars]
    order = [var for sub_list in var_list for var in sub_list]

    # Construct table from results
    table = su2.summary_col(
        res_list,
        model_names=names,
        stars=True,
        regressor_order=order
        )
    
    # Save Table to Text File
    if connected:
        file = '../figures/regression/regression_team_is_nbhd_connected.txt'
    else:
        file = '../figures/regression/regression_team_is_nbhd_all.txt'
    with open(file, 'w') as output:
        output.write(table.as_text())
        
def save_slim_results(results, connected=True):
    
    # Model2OutcomeVariable2Descriptors = slim
    slim = {}
    
    # Iterate through each regression result in results
    for index, regression in enumerate(results.values()):
        
        # Get results object
        execset = regression['execset']
        model = regression['model']
        outcome = regression['out_var']
        res = regression['results']
        
        # Save regression conditions
        slim['execset'] = execset
        slim['model'] = model
        slim['outcome'] = outcome
        
        # Rename log10(run_step + 1) rows and save
        rename = {'log10(run_step + 1)': 'log10(run_step)'}
        slim['params'] = res.params.rename(rename)
        slim['pvalues'] = res.pvalues.rename(rename)
        slim['HC2_se'] = res.HC2_se.rename(rename)
        
        # Populate slim results dictionary from regression
        slim['outcome_log'] = regression['out_log']
        slim['aic'] = res.aic
        slim['bic'] = res.bic
        slim['rsquared'] = res.rsquared
        slim['rsquared_adj'] = res.rsquared_adj
        slim['ess'] = res.ess
        slim['fvalue'] = res.fvalue
        slim['f_pvalue'] = res.f_pvalue
        slim['condition_number'] = res.condition_number
        slim['num_obs'] = res.nobs
    
    # Save result to file
    loc = f'../data/regression/execset{execset:03}_model{model}_'
    if connected:
        fname = 'reg_team_is_nbhd_connected_slim.pickle'
    else:
        fname = 'reg_team_is_nbhd_all_slim.pickle'
    pd.to_pickle(slim, loc + fname)
    
    return slim
    

def load_slim_results(filename):
    return pd.read_pickle(filename)
 
        
#%% Variable builders

def get_drop_cols():
    '''Define columns to drop because not used in regression'''
    
    return ['team_graph_type', 'team_fn_type', 'team_fn_diff_integral',
            'team_fn_alignment', 'team_fn_interdep', 'team_fn_weight',
            'team_fn_frequency', 'team_fn_exponent', 'nbhd_fn']

def get_graph_vars(connected=True):
    
    if connected:
        graph_vars = [
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
    else:
        graph_vars = [
            'team_graph_centrality_degree_mean',
            'team_graph_centrality_degree_stdev',
            'team_graph_centrality_eigenvector_mean',
            'team_graph_centrality_eigenvector_stdev',
            'team_graph_centrality_betweenness_mean',
            'team_graph_centrality_betweenness_stdev',
            'team_graph_nearest_neighbor_degree_mean',
            'team_graph_nearest_neighbor_degree_stdev',
            'team_graph_clustering',
            # 'team_graph_assortativity',  ##
            # 'team_graph_pathlength',      ## Not valid for unconnected graphs
            # 'team_graph_diameter'        ##
            ]
    return graph_vars


#%% Regression Functions

def run_regressions(time, results, index, execset, model, out_var, out_log,
                    connected=True):
    '''Run regression on points.'''
        
    # Load data
    loc = '../data/sets/'
    filename = f'execset{execset:03}_model{model}_team_is_nbhd.pickle'
    df = pd.read_pickle(loc + filename)
    time.update('Dataframe loaded')
            
    # Normalize data
    df = normalize_df(df, out_var, out_log, connected)
    time.update('Dataframe normalized')
    
    # Construct variable equations and results list
    equation = build_equation(out_var, out_log, connected)
    time.update('Equation built')
            
    # Get data for regression
    y, X = dmatrices(equation, data=df, return_type='dataframe')
    time.update('Design matrices built')
    
    # Run regression with matrices
    res = sm.OLS(y, X).fit(
        cov_type='HC2',
        parallel_method='joblib'
        )
    time.update('Regression results generated')

    # Add result to list
    results = add_to_dict(results, res, index, execset=execset, model=model,
                          out_var=out_var, out_log=out_log)
        
    return time, results


#%% Running code

def run_regressions_all(execset, model, connected=True):
    
    # Create timer
    time = Time()
    time.begin('all', 'regressions', 'models')
    
    # Run regressions
    results = {}
    index = 0
    time, results = run_regressions(time, results, index, execset, model,
                                    connected)
    time.update(f'Regression {index+1} complete')
        
    # Save results
    slim = save_slim_results(results, connected)
    generate_table(results, connected)
    time.end('Regression','models')
    
    return slim


if __name__ == '__main__':
    
    slim = run_regressions_all(execset=9, model='3xx')
    
