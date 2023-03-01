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
# ['team_size', 'team_graph_type', 'team_fn_type', 'run_step',
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
#  'team_fn_difficulty',
#  'team_fn_alignment',
#  'team_fn_interdep',
#  'team_performance',
#  'team_productivity',
#  'team_fn_weight',
#  'team_fn_frequency',
#  'team_fn_exponent',
#  'team_fn',
#  'nbhd_fn']


#%% Regression Functions

def run_regressions(results, index, model, run_log, out_var, out_log,
                    connected=True):
    '''Run regression on points.'''
        
    # Load data
    loc = get_loc()
    filename = f'{loc}model{model}_team_is_nbhd.pickle'
    df = pd.read_pickle(filename)
            
    # Normalize data
    df = normalize_df(df, out_var, out_log, connected)
    
    # Construct variable equations and results list
    equation = build_equation(out_var, out_log, connected)
            
    # Get data for regression
    y, X = dmatrices(equation, data=df, return_type='dataframe')
    
    # Run regression with matrices
    res = sm.OLS(y, X, hasconst=True).fit(
        cov_type='HC2',
        parallel_method='joblib'
        )

    # Add result to list
    results = add_to_dict(results, res, index, model=model,
        run_log=run_log, out_var=out_var, out_log=out_log)
        
    return results


#%% Regression utilities

def normalize_df(df, out_var, out_log, connected=True):
    
    norm_cols = get_graph_vars(connected)
    if out_log != 'linear': norm_cols.append(out_var)
    
    for col in df.columns:
        if col in norm_cols:
            df[col] = normalize(df[col])
            
    return df


def normalize(df_col):
    return (df_col - df_col.mean()) / (df_col.max() - df_col.min())


def build_equation(out_var, out_log='linear', connected=True):
    
    # Construct base variables
    if out_var == 'team_productivity':
        base_vars = ['log10(run_step)','team_fn']
    else:
        base_vars = ['log10(run_step + 1)','team_fn']
    
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
        file = '../figures/regression/regression_by_subfns_connected.txt'
    else:
        file = '../figures/regression/regression_by_subfns_all.txt'
    with open(file, 'w') as output:
        output.write(table.as_text())
        
def save_slim_results(results, connected=True):
    
    # Model2OutcomeVariable2Descriptors = slim
    slim = {
        '3xx': {
            'team_performance': {},
            'team_productivity': {}
            },
        '3xg': {
            'team_performance': {},
            'team_productivity': {}
            }
        }
    
    # Iterate through each regression result in results
    for index, regression in enumerate(results.values()):
        
        # Get results object
        model = regression['model']
        outcome = regression['out_var']
        res = regression['results']
        
        # Rename log10(run_step + 1) rows and save
        rename = {'log10(run_step + 1)': 'log10(run_step)'}
        slim[model][outcome]['params'] = res.params.rename(rename)
        slim[model][outcome]['pvalues'] = res.pvalues.rename(rename)
        slim[model][outcome]['HC2_se'] = res.HC2_se.rename(rename)
        
        # Populate slim results dictionary from regression
        slim[model][outcome]['outcome_log'] = regression['out_log']
        slim[model][outcome]['aic'] = res.aic
        slim[model][outcome]['bic'] = res.bic
        slim[model][outcome]['rsquared'] = res.rsquared
        slim[model][outcome]['rsquared_adj'] = res.rsquared_adj
        slim[model][outcome]['ess'] = res.ess
        slim[model][outcome]['fvalue'] = res.fvalue
        slim[model][outcome]['f_pvalue'] = res.f_pvalue
        slim[model][outcome]['condition_number'] = res.condition_number
        slim[model][outcome]['num_obs'] = res.nobs
    
    # Save result to file
    if connected:
        fname = '../data/regression/regression_by_subfns_connected_slim.pickle'
    else:
        fname = '../data/regression/regression_by_subfns_all_slim.pickle'
    pd.to_pickle(slim, fname)
    
    return slim
    

def load_slim_results(filename):
    return pd.read_pickle(filename)
 
        
#%% Variable builders

def get_drop_cols():
    '''Define columns to drop because not used in regression'''
    
    return ['team_graph_type', 'team_fn_type', 'team_fn_difficulty',
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


#%% Running code

def get_loc():
    return '../data/sets/'

def get_iterator():
    iterator = (
        ('3xx', 'fixed_by_fn_subtype', 'team_performance', 'linear'),
        ('3xx', 'fixed_by_fn_subtype', 'team_productivity',  10),
        ('3xg', 'fixed_by_fn_subtype', 'team_performance', 'linear'),
        ('3xg', 'fixed_by_fn_subtype', 'team_productivity',  10),
        )
    return iterator
    

def run_regressions_all(connected):
    
    # Regressions to run
    iterator = get_iterator()
    
    # Create timer
    time = Time()
    time.begin('all', 'regressions', 'models')
    
    # Run regressions
    results = {}
    for index, (model, run_log, out_var, out_log) in enumerate(iterator):
        results = run_regressions(
            results, index, model, run_log, out_var, out_log, connected
            )
        time.update(f'Regression {index+1} of {len(iterator)} complete')
        
    # Save results
    slim = save_slim_results(results, connected)
    generate_table(results, connected)
    time.end('Regression','models')
    
    return slim

def run_regression_test():
    results = run_regressions(['3xg'], ['log'], ['team_productivity'], ['log'])
    generate_table(results)

if __name__ == '__main__':
    
    results_connected = run_regressions_all(connected=True)
    results_all = run_regressions_all(connected=False)
    # results = run_regression_test()
    
