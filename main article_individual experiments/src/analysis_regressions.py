# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 23:05:12 2022

@author: John Meluso
"""

#%% Import libraries

# Import libraries
from itertools import product as pr
from numpy import log10
import pandas as pd
from patsy import dmatrices
from patsy import ModelDesc
from patsy import Term
from patsy import EvalFactor as EvalFactor
from patsy import LookupFactor as LookupFactor
import statsmodels.api as sm
# import statsmodels.formula.api as smf
import statsmodels.iolib.summary2 as su2
# import statsmodels.iolib.smpickle as smp

# Import src files
from classes.Time import Time
import util.analysis as ua


#%% Regression utilities

def normalize_df(df, out_var, out_log, connected=True):

    norm_cols = ua.get_graph_vars(connected)
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


#%% Saving functions

def generate_table(results, props):

    # Construct model names and results
    names = []
    res_list = []
    for index, (result, props) in enumerate(zip(results.values(), props.values())):

        model = str(index+1)
        if props['connected']:
            conn = 'connected metrics'
        else:
            conn = 'all metrics'
        if props['difficulties']:
            diff = 'fn metrics'
        else:
            diff = 'fn fixed effects'
        names.append(f'M{model} ({conn},{diff})')
        res_list.append(result)

    # Build variable order for table
    base_vars = [
        'Intercept',
        'log10(run_step + 1)',
        'team_size',
        'team_size ** 2',
        ]
    graph_vars = ua.get_graph_vars()
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
    fname = '../data/analysis/reg_table.txt'
    with open(fname, 'w') as output:
        output.write(table.as_text())

def save_slim_all(results, props):

    # Model2OutcomeVariable2Descriptors = slim
    slim = {}

    # Iterate through each regression result in results
    for index, (res, props) in enumerate(zip(results.values(), props.values())):

        # Create slim dict for each index entry
        slim[index] = {}

        # Rename log10(run_step + 1) rows and save
        rename = {'log10(run_step + 1)': 'log10(run_step)'}
        slim[index]['params'] = res.params.rename(rename)
        slim[index]['pvalues'] = res.pvalues.rename(rename)
        slim[index]['HC2_se'] = res.HC2_se.rename(rename)

        # Populate slim results dictionary from regression
        slim[index]['aic'] = res.aic
        slim[index]['bic'] = res.bic
        slim[index]['rsquared'] = res.rsquared
        slim[index]['rsquared_adj'] = res.rsquared_adj
        slim[index]['ess'] = res.ess
        slim[index]['fvalue'] = res.fvalue
        slim[index]['f_pvalue'] = res.f_pvalue
        slim[index]['condition_number'] = res.condition_number
        slim[index]['num_obs'] = res.nobs

    # Save result to file
    fname = '../data/analysis/reg_slim.pickle'
    pd.to_pickle(slim, fname)

    return slim

def save_slim_result(index, result, props):
    
    # Create slim dict for each index entry
    slim = {}
    
    # Add props to slim
    for key, value in props.items(): slim[key] = value

    # Rename log10(run_step + 1) rows and save
    rename = {'log10(run_step + 1)': 'log10(run_step)'}
    slim['params'] = result.params.rename(rename)
    slim['pvalues'] = result.pvalues.rename(rename)
    slim['HC2_se'] = result.HC2_se.rename(rename)

    # Populate slim resultsults dictionary from regression
    slim['aic'] = result.aic
    slim['bic'] = result.bic
    slim['rsquared'] = result.rsquared
    slim['rsquared_adj'] = result.rsquared_adj
    slim['ess'] = result.ess
    slim['fvalue'] = result.fvalue
    slim['f_pvalue'] = result.f_pvalue
    slim['condition_number'] = result.condition_number
    slim['num_obs'] = result.nobs

    # Save slim pickle
    fname = f'../data/analysis/reg_slim_{index:02}.pickle'
    pd.to_pickle(slim, fname)
    
    # Save full result to pickle
    fname = f'../data/analysis/reg_full_{index:02}.pickle'
    pd.to_pickle(result, fname)
    

def load_slim_results(filename):
    return pd.read_pickle(filename)


#%% Variable builders

def get_fn_vars(controls='both'):
    
    if controls == 'metrics':
        fn_vars = [
            'team_fn_diff_integral',
            'log10(team_fn_diff_peaks)',
            'team_fn_alignment',
            'team_fn_interdep'
            ]
    elif controls == 'fes':
        fn_vars = ['C(team_fn)']
    elif controls == 'both':
        fn_vars = [
            'C(team_fn)',
            'team_fn_diff_integral',
            'log10(team_fn_diff_peaks)',
            'team_fn_alignment',
            'team_fn_interdep'
            ]
    else:
        raise RuntimeError(f'Function controls input {controls} is not valid.')
        
    return fn_vars


#%% Equation builders

def build_equations():

    # Connected & function difficulty configurations
    connected_metrics = [True, False] #[0-5],[6-11]
    step_intx = [False, True] #[0-2,6-8],[3-5,9-11]
    fn_controls = ['metrics', 'fes', 'both'] #[0,3,6,9],[1,4,7,10],[2,5,8,11]

    # Iteratively build equations
    equations = {}
    props = {}
    for index, (conn, stix, fn) \
        in enumerate(pr(connected_metrics, step_intx, fn_controls)):

        # Step limit variable
        steps = 'log10(agent_steplim)'

        # Construct base equation
        equation = ModelDesc([Lookup('team_performance')],
                             [Term([]),
                              Eval('log10(run_step + 1)'),
                              Lookup('team_size'),
                              Eval('team_size**2'),
                              Eval(steps)])

        # Add network and function variables
        equation = add_vars(equation, ua.get_graph_vars(conn))
        equation = add_vars(equation, get_fn_vars(fn))
        if stix:
            equation = add_vars_intx(equation, steps, ua.get_graph_vars(conn))
            equation = add_vars_intx(equation, steps, get_fn_vars('metrics'))

        # Save equation to list of equations
        equations[index] = equation

        # Save equation properties
        props[index] = {
            'name': f'M{index:02}\n({conn},{stix},{fn})',
            'connected_metrics': conn,
            'step_interactions': stix,
            'fn_controls': fn
            }

    return equations, props

def add_vars(desc, extra_vars):
    desc.rhs_termlist += [Eval(p) for p in extra_vars]
    return desc

def add_vars_intx(desc, base_var, extra_vars):
    desc.rhs_termlist += [Eval_intx(base_var, p) for p in extra_vars]
    return desc

def Lookup(var):
    return Term([LookupFactor(var)])

def Eval(var):
    return Term([EvalFactor(var)])

def Eval_intx(var1, var2):
    return Term([EvalFactor(var1), EvalFactor(var2)])


#%% Regression models

def run_regression(time, df, equation):

    # Print equation
    equation.describe()

    # Get data for regression
    y, X = dmatrices(equation, data=df, return_type='dataframe')
    time.update('Design matrices built')

    # Run regression with matrices
    result = sm.OLS(y, X).fit(
        cov_type='HC2',
        parallel_method='joblib'
        )
    time.update('Regression results generated')

    return time, result


#%% Running code

def run_all(execset, model):

    # Create timer
    time = Time()
    time.begin('all', 'regressions', 'models')

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

    # Iterate over equations
    for index, equation in equations.items():

        # Run regressions
        time, result = run_regression(time, df, equation)
        time.update(f'Regression {index+1} of {len(equations)} complete')

        # Save result
        save_slim_result(index, result, props[index])
        
        # delete result to clear memory
        del result

    # End regression
    time.end('Regression','models')

    return slim


if __name__ == '__main__':

    slim = run_all(execset=10, model='3xx')
