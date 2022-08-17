# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 11:05:44 2022

@author: John Meluso
"""

#%% Import libraries

# Import libraries
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.smpickle import save_pickle

# Import src files
from classes.Time import Time



#%% Regression Functions

def regress_points(model, mode, out_var, out_log):
    '''Run regression on points.'''

    # Start timer
    time = Time()
    time.begin('Model', f'{mode} regression on {out_var}', model)

    # Load data
    filename = f'../data/sets/model{model}.parquet'
    X = pd.read_parquet(filename)

    # Construct variable equations and results list
    y = X[out_var]
    X.drop(columns=out_var, inplace=True)
    time.update('Equation constructed')

    # Run dataframe through patsy
    time.update('Design matrices constructed')

    # Create OLS model
    results = sm.OLS(y,X).fit(cov_type='HC2').summary()
    print(results.summary())
    save_pickle(results, f'../data/regression/model{model}.pickle')
    time.end(f'{mode} regression on {out_var}', model)


#%% Running code

if __name__ == '__main__':
    
    regress_points('3xx', 'fixed_by_fn_type', 'team_performance', None)
