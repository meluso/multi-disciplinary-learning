# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 14:16:03 2021

@author: John Meluso
"""

def set_functions(fns):
    '''Sets the specified functions to their default options.'''
    
    functions = {}
    
    for fn in fns:
        functions[fn] = eval('set_' + fn + '()')
    
    return functions


def set_all_functions_off():
    '''Returns 'na' as the only function for a model where this function is not
    applicable, such as the Neighborhood function in a model which does not
    include neighborhoods.'''
    
    return {'na': set_na()}


def set_all_functions_default():
    '''Sets all functions to their default options.'''
    
    all_functions = {
        'average': set_average(),
        'sphere': set_sphere(),
        'root': set_root(),
        'sin2': set_sin2(),
        'sin2sphere': set_sin2sphere(),
        'sin2root': set_sin2root(),
        'losqr_hiroot': set_losqr_hiroot(),
        'hisqr_loroot': set_hisqr_loroot(),
        'max': set_max(),
        'min': set_min(),
        'median': set_median(),
        'kth_power': set_kth_power(),
        'kth_root': set_kth_root(),
        'ackley': set_ackley()
        }
    
    return all_functions


def set_all_weight_node():
    '''Sets all functions to their default options.'''
    
    wts = ['node']
    
    all_functions = {
        'average': set_average(wts),
        'sphere': set_sphere(wts),
        'root': set_root(wts),
        'sin2': set_sin2(wts),
        'sin2sphere': set_sin2sphere(wts),
        'sin2root': set_sin2root(wts),
        'losqr_hiroot': set_losqr_hiroot(wts),
        'hisqr_loroot': set_hisqr_loroot(wts),
        'max': set_max(),
        'min': set_min(),
        'median': set_median(),
        'kth_power': set_kth_power(wts),
        'kth_root': set_kth_root(wts),
        'ackley': set_ackley()
        }
    
    return all_functions


def set_agent_passthrough():
    '''Sets agent functions to pass through their values by using the average
    function, which passes the value as is when implemented at agent level.'''
    
    return {'average': set_average(wts=['node'])}


def set_average(wts=['node','degree']):
    '''Set options for average function.'''
    return set_fn_opts('average', wts=wts)


def set_sphere(wts=['node','degree']):
    '''Set options for sphere function.'''
    return set_fn_opts('sphere', wts=wts)


def set_root(wts=['node','degree']):
    '''Set options for root function.'''
    return set_fn_opts('root', wts=wts)


def set_sin2(wts=['node','degree'],freq=['uniform','degree']):
    '''Set options for sin2 function.'''
    return set_fn_opts('sin2', wts=wts, freq=freq)


def set_sin2sphere(wts=['node','degree'],freq=['uniform','degree']):
    '''Set options for sin2sphere function.'''
    return set_fn_opts('sin2sphere', wts=wts, freq=freq)


def set_sin2root(wts=['node','degree'],freq=['uniform','degree']):
    '''Set options for sin2root function.'''
    return set_fn_opts('sin2root', wts=wts, freq=freq)


def set_losqr_hiroot(wts=['node','degree'],exp=['uniform','degree']):
    '''Set options for losqr_hiroot function.'''
    return set_fn_opts('losqr_hiroot', wts=wts, exp=exp)


def set_hisqr_loroot(wts=['node','degree'],exp=['uniform','degree']):
    '''Set options for hisqr_loroot function.'''
    return set_fn_opts('hisqr_loroot', wts=wts, exp=exp)


def set_max():
    '''Set options for max function.'''
    return set_fn_opts('max')


def set_min():
    '''Set options for min function.'''
    return set_fn_opts('min')


def set_median():
    '''Set options for median function.'''
    return set_fn_opts('median')


def set_kth_power(wts=['node','degree']):
    '''Set options for kth power function.'''
    return set_fn_opts('kth_power', wts=wts)


def set_kth_root(wts=['node','degree']):
    '''Set options for kth root function.'''
    return set_fn_opts('kth_root', wts=wts)


def set_ackley():
    '''Set options for ackley function.'''
    return set_fn_opts('ackley')

def set_na():
    '''Set options for case where function is not applicable.'''
    return set_fn_opts('na')


def set_fn_opts(fn_type, **kwargs):
    '''Set options for a specified function type.'''
    
    opts = {}
    
    if fn_type == 'average' or fn_type == 'sphere' or fn_type == 'root' \
        or fn_type == 'kth_power' or fn_type == 'kth_root':
            
        for key, value in kwargs.items():
            
            if key == 'wts':
                opts['weights'] = value
            else:
                raise_fn_opts_error(key, fn_type)
            
    elif fn_type == 'sin2' or fn_type == 'sin2sphere' or fn_type == 'sin2root':
        
        for key, value in kwargs.items():
            
            if key == 'wts':
                opts['weights'] = value
            elif key == 'freq':
                opts['frequency'] = value
            else:
                raise_fn_opts_error(key, fn_type)
                    
    elif fn_type == 'losqr_hiroot' or fn_type == 'hisqr_loroot':
        
        for key, value in kwargs.items():
            
            if key == 'wts':
                opts['weights'] = value
            elif key == 'exp':
                opts['exponent'] = value
            else:
                raise_fn_opts_error(key, fn_type)
    
    elif fn_type == 'max' or fn_type == 'min' or fn_type == 'median' \
        or fn_type == 'ackley' or fn_type == 'na':
        
        pass
            
    else:
        raise RuntimeError('Function type ' + fn_type + ' is not valid.')
        
    return opts


def get_fn_opts(fn_type,in_tuple):
    '''Returns a dictionary with function options for a single case.'''
    
    fn_opts = {}
    
    if fn_type == 'average' or fn_type == 'sphere' or fn_type == 'root' \
        or fn_type == 'kth_power' or fn_type == 'kth_root':
            
        fn_opts['weight'] = in_tuple[0]
        
    elif fn_type == 'sin2' or fn_type == 'sin2sphere' or fn_type == 'sin2root':
        
        fn_opts['weight'] = in_tuple[0]
        fn_opts['frequency'] = in_tuple[1]
        
    elif fn_type == 'losqr_hiroot' or fn_type == 'hisqr_loroot':
        
        fn_opts['weight'] = in_tuple[0]
        fn_opts['exponent'] = in_tuple[1]
    
    elif fn_type == 'max' or fn_type == 'min' or fn_type == 'median' \
        or fn_type == 'ackley' or fn_type == 'na':
        
        pass
            
    else:
        raise RuntimeError('Function type ' + fn_type + ' is not valid.')
        
    return fn_opts

def raise_fn_opts_error(key,fn_type):
    '''Raises an error for the given key and function type.'''
    raise RuntimeError('Input ' + key + ' invalid for ' + fn_type \
                       + ' function.')