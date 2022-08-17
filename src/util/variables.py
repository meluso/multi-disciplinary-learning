# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 15:18:40 2022

@author: John Meluso
"""

import classes.Variables as Variables

def create_variables():
    all_vars = [
        # Create model parameters
        'model_type',
        
        # Create team parameters
        'team_size',
        'team_graph_type',
        'team_graph_opts',
            'team_graph_k',
            'team_graph_m',
            'team_graph_p',
        'team_fn_type',
        'team_fn_opts',
            'team_fn_exponent',
            'team_fn_frequency',
            'team_fn_weight',
        
        # Create neighborhood parameters
        'nbhd_fn_type',
        'nbhd_fn_opts',
            'nbhd_fn_exponent',
            'nbhd_fn_frequency',
            'nbhd_fn_weight',
        
        # Create agent parameters
        'agent_fn_type',
        'agent_fn_opts',
            'agent_fn_exponent',
            'agent_fn_frequency',
            'agent_fn_weight',
        'agent_steplim',
        
        # Create simulation paraemters
        'num_steps',
        'run_ind',
        
        # Create running values
        'run_step',
        
        # Create graph descriptors
        'team_graph_centrality_degree_mean',
        'team_graph_centrality_degree_stdev',
        'team_graph_centrality_eigenvector_mean',
        'team_graph_centrality_eigenvector_stdev',
        'team_graph_centrality_betweenness_mean',
        'team_graph_centrality_betweenness_stdev',
        'team_graph_nearest_neighbor_degree_mean',
        'team_graph_nearest_neighbor_degree_stdev',
        'team_graph_clustering',
        'team_graph_density',
        'team_graph_assortativity',
        'team_graph_pathlength',
        'team_graph_diameter',
        
        # Create fn descriptors
        'team_fn_diff_integral',
        'team_fn_diff_peaks',
        'team_fn_alignment',
        'team_fn_interdep',
        'nbhd_fn_diff_integral',
        'nbhd_fn_diff_peaks',
        'agent_fn_diff_integral',
        'agent_fn_diff_peaks',
        
        # Create output metrics
        'team_performance',
        'team_productivity'
        ]
    
    variables = dict()
    for v in all_vars:
        variables[v] = get_variable(v)
    return variables


### Analysis Variable Functions ############################################
    
    # Variable attributes ###
    # dtype: type = object
    # param_model: bool = False
    # param_sim: bool = False
    # descriptor: bool = False
    # running: bool = False
    # outcome: bool = False
    # index: bool = False
    # levels: Levels = Levels()
    # default_slice: object = None
    
    # Level attributes ###
    # team: bool = False
    # nbhd: bool = False
    # agent: bool = False
    
def get_all_of(attribute, **kwargs):
    '''Gets a list of all the variables with a specified attribute.'''
    att_dict = {
        key: [] for (key, value) in variables.items()
        if getattr(value, attribute)
        }
    
    # Fill with any keyword arguments
    for key, value in kwargs.items():
        att_dict[key] = value
        
    return att_dict

def get_attribute_if_any(attribute1, attribute_list: list):
    '''Gets a dictionary of all variables with attribute 1 and any attributes
    in the list.'''
    return {
        key: getattr(value, attribute1) for (key, value) in variables.items() 
        if any([getattr(value, attribute2) for attribute2 in attribute_list])
        }
        
def get_attribute_if_all(attribute1, attribute_list: list, output=dict):
    '''Gets a list of all variables with attribute 1 and all attributes in the
    list.'''
    return {
        key: getattr(value, attribute1) for (key, value) in variables.items() \
        if all([getattr(value, attribute2) for attribute2 in attribute_list])
        }
        
def get_any_levels(attribute, levels_list: list):
    '''Gets a list of all variables with an attribute and that level.'''
    return [
        key for (key, value) in variables.items() if getattr(value, attribute)
        and any([getattr(value.levels, level) for level in levels_list])
        ]
        
def get_not_levels(attribute, levels_list: list):
    '''Gets a list of variables with an attribute and not that level.'''
    return [
        key for (key, value) in variables.items() if getattr(value, attribute)
        and not any([getattr(value.levels, level) for level in levels_list])
        ]

def get_dtypes(attribute_list=['index','descriptor','outcome']):
    '''Returns a datatype dictionary to dask for dask reading.'''
    return get_attribute_if_any('dtype',attribute_list)

def get_param_sim(**kwargs):
    '''Returns the parameter variables.'''
    return get_all_of('param_sim', **kwargs)

def get_param_model():
    '''Returns the parameter variables.'''
    return get_all_of('param_model')

def get_descriptors():
    '''Returns the descriptive variables.'''
    return get_all_of('descriptor')

def get_runnings():
    '''Returns the running variables.'''
    return get_all_of('running')

def get_outcomes():
    '''Returns the output variables.'''
    return get_all_of('outcome')

def get_indices():
    '''Returns the index variables.'''
    return get_all_of('index')

def get_default_slices():
    '''Returns the default slice for each variable.'''
    return get_attribute_if_any('default_slice',['index'])

def team_indices():
    '''Gets variables that apply to the team level only.'''
    return get_any_levels('index',['team'])
    
def not_team_indices():
    '''Get variables to exclude from team indices.'''
    return get_not_levels('index',['team'])

def team_and_agent_indices():
    '''Gets variables that apply to the team or agent levels.'''
    return get_any_levels('index',['team','agent'])

def not_team_or_agent_indices():
    '''Gets variables that do not apply to the team or agent levels.'''
    return get_not_levels('index',['team','agent'])

def team_and_nbhd_indices():
    '''Gets variables that apply to the team or neighborhood levels.'''
    return get_any_levels('index',['team','nbhd'])

def not_team_or_nbhd_indices():
    '''Gets variables that do not apply to the team or neighborhood levels.'''
    return get_not_levels('index',['team','nbhd'])

def all_level_indices():
    '''Gets variables that apply to the team or neighborhood levels.'''
    return get_any_levels('index',['team','nbhd','agent'])

def not_level_indices():
    '''Gets variables that do not apply to the team or neighborhood levels.'''
    return get_not_levels('index',['team','nbhd','agent'])

def get_variable(vrbl):
    '''Selects and returns the variable of the correct type.'''
    try: return getattr(Variables,vrbl.capitalize())()
    except: raise RuntimeError(f'Variable type {vrbl} is not valid.')

### Create Variables #######################################################

variables = create_variables()