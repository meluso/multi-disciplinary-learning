# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:35:33 2022

@author: John Meluso
"""

#%% Import libraries

# Import libraries
from numpy import log10
import pandas as pd
from patsy import dmatrices
import pickle
import statsmodels.api as sm
import statsmodels.iolib.summary2 as su2

# Import src files
from classes.Time import Time

# Columns loaded from pickle:
# ['team_size', 'team_graph_type', 'team_fn_type', 'run_step',
#  'team_graph_centrality_degree_mean',
#  'team_graph_centrality_degree_stdev',
#  'team_graph_centrality_eigenvector_mean',
#  'team_graph_centrality_eigenvector_stdev',
#  'team_graph_centrality_betweenness_mean',
#  'team_graph_centrality_betweenness_stdev',
#  'team_graph_nearest_neighbor_degree_mean',
#  'team_graph_nearest_neighbor_degree_stdev', 'team_graph_clustering',
#  'team_graph_assortativity', 'team_graph_pathlength',
#  'team_graph_diameter', 'team_fn_difficulty', 'team_fn_alignment',
#  'team_fn_interdep', 'team_performance', 'team_productivity',
#  'team_fn_weight', 'team_fn_frequency', 'team_fn_exponent', 'team_fn',
#  'nbhd_fn']


#%% Regression Functions

def run_regressions(index, model, run_log, out_var, out_log, time):
    '''Run regression on points.'''
        
    # Load data
    loc = get_loc()
    filename = f'{loc}model{model}.pickle'
    df = pd.read_pickle(filename)
    time.update('Data loaded')
    
    # Construct variable equations and results list
    equation = build_equation(out_var, out_log)
    time.update('Equation built')
            
    # Get data for regression
    y, X = dmatrices(equation, data=df, return_type='dataframe')
    time.update('Design matrices constructed')
    
    # Run regression with matrices
    res = sm.OLS(y, X).fit(
        cov_type='HC2',
        parallel_method='joblib'
        )
    print(res.summary())
    time.update('Regression fitting complete')
    
    # Save results to pickle
    time.update('Results saved to pickle')
    
    return res, time


#%% Regression utilities

def normalize_df(df, out_var, out_log):
    
    norm_cols = get_graph_vars()
    if out_log != 'linear': norm_cols.append(out_var)
    
    for col in df.columns:
        if col in norm_cols:
            df[col] = normalize(df[col])
            
    return df


def normalize(df_col):
    return (df_col - df_col.mean()) / (df_col.max() - df_col.min())


def build_equation(out_var, out_log='linear'):
    
    # Construct base variables
    if out_var == 'team_productivity':
        base_vars = ['log10(run_step)','team_fn','nbhd_fn']
    else:
        base_vars = ['log10(run_step + 1)','team_fn','nbhd_fn']
    
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
    for var in get_graph_vars():
        equation += ' + ' + var
        
    return equation


def add_to_dict(results_dict, results, index, **kwds):
    
    # Add results to index
    results_dict[index] = dict(results = results)
    
    # Add other keywords to index
    for key, value in kwds.items():
        results_dict[index][key] = value

    return results_dict


def generate_table(results):
    
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
    graph_vars = get_graph_vars()
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
    file = '../figures/regression/regression_by_subfns_all.txt'
    with open(file, 'w') as output:
        output.write(table.as_text())
        
def save_slim_results(model, out_var, out_log, res):
    
    # Model2OutcomeVariable2Descriptors = slim
    res_dict = {}
    
    # Rename log10(run_step + 1) rows and save
    rename = {'log10(run_step + 1)': 'log10(run_step)'}
    res_dict['params'] = res.params.rename(rename)
    res_dict['pvalues'] = res.pvalues.rename(rename)
    res_dict['HC2_se'] = res.HC2_se.rename(rename)
    
    # Populate slim results dictionary from regression
    res_dict['outcome_log'] = out_log
    res_dict['aic'] = res.aic
    res_dict['bic'] = res.bic
    res_dict['rsquared'] = res.rsquared
    res_dict['rsquared_adj'] = res.rsquared_adj
    res_dict['ess'] = res.ess
    res_dict['fvalue'] = res.fvalue
    res_dict['f_pvalue'] = res.f_pvalue
    res_dict['condition_number'] = res.condition_number
    res_dict['num_obs'] = res.nobs
    
    return res_dict
    

def load_slim_results(filename):
    return pd.read_pickle(filename)
 
        
#%% Variable builders

def get_drop_cols():
    '''Define columns to drop because not used in regression'''
    
    return ['team_graph_type', 'team_fn_type', 'team_fn_difficulty',
            'team_fn_alignment', 'team_fn_interdep', 'team_fn_weight',
            'team_fn_frequency', 'team_fn_exponent', 'nbhd_fn']

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


#%% Running code

def get_loc():
    return '/gpfs2/scratch/jmeluso/ci-greedy-agents-base/data/sets/'

def get_iterator():
    iterator = (
        ('3xx', 'fixed_by_fn_subtype', 'team_performance', 'linear'),
        ('3xx', 'fixed_by_fn_subtype', 'team_productivity',  10),
        ('3xg', 'fixed_by_fn_subtype', 'team_performance', 'linear'),
        ('3xg', 'fixed_by_fn_subtype', 'team_productivity',  10),
        )
    return iterator
    

def run_regressions_all():
    
    # Regressions to run
    iterator = get_iterator()
    
    # Create timer
    time = Time()
    time.begin('all', 'regressions', 'models')
    
    # Create results dictionary
    results = []  
    
    for index, (model, run_log, out_var, out_log) in enumerate(iterator):
        
        # Run regression
        time.update(f'Starting regression {index+1} of {len(iterator)}')
        res, time = run_regressions(index, model, run_log, out_var, out_log, time)
        time.update(f'Regression {index+1} of {len(iterator)} complete')
        
        # Save regression
        results.append(save_slim_results(model, out_var, out_log, res))
        generate_table(res)
        
    # Save slim regression results
    with open(f'{get_loc()}model{model}_regression.pickle') as file:
        pickle.dump(results, file)
    
        

def run_regression_test():
    results = run_regressions(['3xg'], ['log'], ['team_productivity'], ['log'])
    generate_table(results)

    

if __name__ == '__main__':
    
    run_regressions_all()
    # results = run_regression_test()
    
    
