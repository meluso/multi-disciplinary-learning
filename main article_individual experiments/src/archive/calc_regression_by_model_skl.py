# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:05:44 2022

@author: John Meluso
"""

#%% Import libraries

# Import libraries
import dask.dataframe as dd
from dask_ml.linear_model import LinearRegression

# Import source files
import util.analysis as ua


#%% Regression Functions

def regress_points(model, mode, out_var, out_log):
    '''Run regression on points.'''

    # Load data
    loc = f'../data/sets/model{model}.parquet'
    df = dd.read_parquet(loc)

    # Construct variable equation and results list
    y = df[out_var].to_dask_array(lengths=True)
    X = df.drop(columns=out_var).to_dask_array(lengths=True)

    # Create OLS model
    lm = LinearRegression(fit_intercept=True)
    results = lm.fit(X,y)
    lm.score(X,y)
    
    # # Save Results to Pickle
    loc = '../data/regression/'
    file = f'{loc}reg_{out_log}_{out_var}_{mode}_model{model}.pickle'
    return results    

#%% Running code

def run_regression(model, mode, out_var, out_log=None):
    fn = lambda: regress_points(model, mode, out_var, out_log)
    results = ua.run_dask_cluster(fn, 6)
    return results

if __name__ == '__main__':
    
    results = run_regression('3xx','fixed_by_fn_subtype','team_productivity')
    # run_regression('3xx','fixed_by_fn_subtype','team_performance')
