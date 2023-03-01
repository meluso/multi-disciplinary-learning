# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:02:36 2021

@author: John Meluso
"""

# Import model functions
import classes.Parameters as Parameters


def count_params(pset):
    '''Counts the number of cases in the set of parameters.'''
    
    # Get parameter object
    ParamObject = call_params_object(pset)
    
    # Retreive parameters
    count, __ = ParamObject.build_params()
    
    return count + 1


def get_params(pset,all_cases=False,get_cases=[]):
    '''Gets the cases specified in the set of parameters.'''
    
    # Get parameter object
    ParamObject = call_params_object(pset)
    
    # Retreive parameters
    __, params = ParamObject.build_params(
        all_cases=all_cases,
        get_cases=get_cases
        )
    
    if len(params) == 1:
        return params[0]
    else:
        return params
    

def count_and_get_params(pset,all_cases=False,get_cases=[]):
    '''Gets the cases specified in the set of parameters.'''
    
    # Get parameter object
    ParamObject = call_params_object(pset)
    
    # Retreive parameters
    count, params = ParamObject.build_params(
        all_cases=all_cases,
        get_cases=get_cases
        )
    
    if len(params) == 1:
        return count + 1, params[0]
    else:
        return count + 1, params
    

def call_params_object(pset):
    '''Calls the parameter object given an input pset.'''
    try:
        ParamObject = getattr(Parameters, f'Params{pset:03}')()
    except ValueError:
        try:
            ParamObject = getattr(Parameters, f'Params{pset}')()
        except:
            raise RuntimeError(f'Input {pset} is not a valid parameter set.')
    return ParamObject