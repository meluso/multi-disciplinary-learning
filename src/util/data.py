# -*- coding: utf-8 -*-
'''
Created on Thu Oct 28 17:49:34 2021

@author: John Meluso
'''

# Import libraries
import csv
import json
import pickle
import sys
    
    
def get_graph_opts(prefix='team'):
    names = ['k','m','p']
    opts = {}
    for name in names: opts[prefix + '_graph_' + name] = []
    return names, opts


def get_fn_opts(prefix):
    names = ['weight','frequency','exponent']
    opts = {}
    for name in names: opts[prefix + '_fn_' + name] = []
    return names, opts


def manage_results(mode,points,dir_prefix):
    '''Select the data format(s) to get for the execution(s).'''
    
    # Manage data based on run mode
    if mode == 'save' or mode == 'both':
        save_json(dir_prefix,points)
    result = points.get_dataframe()
    return result


def get_directory(paramset, partition=1, output=False):
    '''Gets the directory either locally or on cluster.'''
    
    # Build execution string & check validity
    if output:
        pset_str = 'sets'
    elif paramset == 'test':
        pset_str = 'test'
    elif isinstance(paramset, int):
        pset_str = f'exec{paramset:03}'
    else: raise RuntimeError('Paramset ' + paramset + ' is invalid.')
    
    # Set location based on platform
    if sys.platform.startswith('linux'):
        if partition == 2:
            prefix = '/gpfs2/scratch/jmeluso'
        else:
            prefix = '/gpfs1/home/j/m/jmeluso'
        
        loc = prefix + '/ci-greedy-agents-base/data/' + pset_str + '/'
    else:
        if partition == 2:
            prefix = 'D:/Data/ci-greedy-agents-base'
        else:
            prefix = '..'
        loc = prefix + '/data/' + pset_str + '/'
    return loc
    

def get_test_loc():
    '''Sets test variables for test saving and loading.'''

    # Set location based on platform
    if sys.platform.startswith('linux'):
        loc = '~/ci-greedy-agents-base/data/test/'
    else:
        loc = '../data/test/'
    return loc + 'test'


def save_test(points):
    '''Saves data from a test run.'''

    # Get test variables
    dir_prefix = get_test_loc()

    # Write run summary and history to location
    save_json(dir_prefix, points)


def load_test():
    '''Loads data from a test run.'''

    # Get test variables
    dir_prefix = get_test_loc()

    # Read run summary and history to variables
    return load_json(dir_prefix)


def save_pickle(dir_prefix, points):
    '''Save data from a set of model runs.'''
    
    # Get object in list form
    results = points.get_dict()
    
    # Write run history to location
    with open(dir_prefix + '.pickle','wb') as file:
        pickle.dump(results, file)


def load_pickle(filename):
    '''Loads data from a set of model runs.'''

    # Read run summary to variable
    with open(filename,'rb') as file:
        points = pickle.load(file)
        
    return points


def save_json(dir_prefix, points):
    '''Save data from a set of model runs.'''
    
    # Get object in list form
    results = points.get_dict()
    
    # Write run history to location
    with open(dir_prefix + '.json','w') as file:
        json.dump(results, file)


def load_json(filename):
    '''Loads data from a set of model runs.'''

    # Read run summary to variable
    with open(filename  + '.json','r') as file:
        points = json.load(file)
        
    return points


def save_csv(dir_prefix, points):
    '''Save points to csv.'''
    
    # Get object in list form
    results = points.get_dict()
    
    # Write run history to location
    with open(dir_prefix + '.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(results.keys())
        writer.writerows(zip(*results.values()))
        

def load_csv(filename):
    '''Load points from csv.'''
    
    # Read run summary to variable
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        summary = next(reader)
        
    return summary
        

def get_exec_num():
    '''Gets execution number from system inputs.'''
    if sys.platform.startswith('linux'):
        return sys.argv[1]
    else:
        return 0