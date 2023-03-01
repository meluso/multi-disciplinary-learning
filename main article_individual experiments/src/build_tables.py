# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 20:38:59 2022

@author: jam
"""

# Import libraries
import itertools as it
import pandas as pd

# Import source files
import fig_settings as fs
import util.data as ud
import util.plots as up
import util.variables as uv

All = slice(None)
fs.set_fonts()

def build_network_measures_table(execset=10):
    
    # Get execset data
    file_in = f'../data/sets/execset{execset:03}_stats.pickle'
    params2stats = ud.load_pickle(file_in)
    
    # Build row indices
    # name = ['team_performance','team_productivity']
    name = All
    stat = 'mean'
    cols = (name, stat)
    var2slice = get_var2slice()
    
    # Slice dataset down to the following fields:
    # Model | Graph | Team Fn | Nbhd Fn | name
    df = params2stats.loc[tuple(var2slice.values()),cols].reset_index()
    df = df.droplevel(1,axis=1)
        
    # Build cumulative variables for fns and graphs
    var_prefixes = ['team_graph']
    for prefix in var_prefixes:
        df = combine_columns(df, prefix)
    
    # Drop all the variables we don't need
    keep_vars = (
        df.columns.str.startswith('team_graph') \
        & ~df.columns.str.fullmatch('team_graph_type') \
        & ~df.columns.str.fullmatch('team_graph_k') \
        & ~df.columns.str.fullmatch('team_graph_m') \
        & ~df.columns.str.fullmatch('team_graph_p') \
        & ~df.columns.str.fullmatch('team_graph_density')
        )
    df = df.loc[:,keep_vars]
    
    # Reorder columns
    df = df[['team_graph'] + [c for c in df if c not in ['team_graph']]]
    
    # Relabel networks
    nets = up.get_graph_labels()
    df = df.replace(nets)
    df = df.set_index('team_graph')
    df.index = df.index.set_names('Network type')
    df = df.sort_index()
    
    # Relabel measures
    measures = up.get_metric_labels()
    df.columns = [measures[col] for col in df.columns]
    
    # fill nans
    df = df.fillna('')
    
    # Print table
    styler = df.style
    styler.applymap_index(
        lambda v: "rot:--rwrap--latex;", axis=1
        )
    styler.format(precision=3)
    print(styler.to_latex(
        hrules=True,
        convert_css=True,
        ))
    
    return df
    
def get_var2slice():
    
    # Build row indices
    var2slice = {key: All for key in uv.get_default_slices().keys()}
    var2slice['model_type'] = ['3xx']
    var2slice['team_size'] = 25
    var2slice['agent_steplim'] = 0.1
    var2slice['run_step'] = 0
    var2slice['team_fn_type'] = 'average'
    var2slice['team_fn_exponent'] = 'na'
    var2slice['team_fn_frequency'] = 'na'
    var2slice['team_fn_weight'] = 'node'
    var2slice['nbhd_fn_type'] = 'average'
    var2slice['nbhd_fn_exponent'] = 'na'
    var2slice['nbhd_fn_frequency'] = 'na'
    var2slice['nbhd_fn_weight'] = 'node'
    del var2slice['run_ind']
    
    return var2slice

def combine_columns(df, prefix):
    cols_with_prefix = ['team_graph_type','team_graph_k','team_graph_m','team_graph_p']
    for col in cols_with_prefix:
        if prefix in df.columns:
            df[prefix] = [x + '_' + str(y) for x, y in zip(df[prefix], df[col])]
        else:
            df[prefix] = df[col]
    return df
        
def build_regression_tables(regs_to_load=[3,4,5,9,10,11]):
    
    metric_labels = up.get_metric_labels()
    other_labels = {
        'log10(run_step)': 'log$_{10}$(Time step + 1)',
        'log10(agent_steplim)': 'log$_{10}$(Search radius)',
        'team_size ** 2': 'Team size$^{2}$',
        'team_size': 'Team size',
        'log10(team_fn_diff_peaks)': 'log$_{10}$(Task num peaks)',
        ':': ' $\\times$ '
        }
    
    # Build of regression columns to combine
    reg_cols = {}
    
    for ii, index in enumerate(regs_to_load):
        
        # Load slim results
        fname = f'../data/analysis/reg_slim_{index:02}.pickle'
        reg = pd.read_pickle(fname)
        
        # Build table
        df = pd.concat(
            [reg['params'],reg['pvalues'],reg['HC2_se']],
            axis=1,
            )
        df.columns=['params','pvalues','HC2_se']
        
        # Combine columns
        df = df.apply(build_coefficients, axis=1)
        
        # Replace index names
        for key in metric_labels.keys():
            df.index = df.index.str.replace(key, metric_labels[key], regex=False)
        for key in other_labels.keys():
            df.index = df.index.str.replace(key, other_labels[key], regex=False)
        
        # Remove fixed effects
        df = df.loc[~df.index.str.contains('C(team_fn)',regex=False),'model']
        
        # Build a dataframe of statistics
        head, foot = build_head_foot(reg)
        
        # Add the header and footer
        column = pd.concat([head, df, foot])
        
        # Add the column to the dictionary
        reg_cols[f'Model {ii+1}'] = column[0]
        
    # Make the dictionary into a dataframe
    table_columns = [col for col in reg_cols.values()]
    regtable = pd.concat(table_columns, axis=1)
    regtable.columns = [model for model in reg_cols.keys()]
    
    # Make multiindex
    regtable.index = pd.MultiIndex.from_tuples(build_multiindex(regtable.index))
    
    # Fill na values with an empty string
    regtable = regtable.fillna('')
        
    # Write table in two groups
    col_grps = [
        ['Model 1','Model 2','Model 3'],
        ['Model 4','Model 5','Model 6']
        ]
    for col_grp in col_grps:
        print(regtable.loc[:,col_grp].style.to_latex(
            hrules=True,
            column_format='llccc',
            sparse_index=True,
            clines='skip-last;data'
            ))
        
    return regtable
        
def build_coefficients(df):
    
    # Get regression coefficients
    param = f'{df["params"]:.3}'
    
    # Build regression coefficient pvalues stars
    pstar = star_maker(df['pvalues'])
    
    # Get regression standard errors
    stderr = f' ({df["HC2_se"]:.3})'
    
    # Combine into single output
    df['model'] = param + pstar + stderr
    
    return df

def build_head_foot(reg):
    
    # Header fields to get, in order
    head = {
        'fn_controls': {
            'index': 'Task controls',
            'value': which_task_controls(reg["fn_controls"]),
            },
        'connected_metrics': {
            'index': 'Network measures',
            'value': which_network_measures(reg["connected_metrics"]),
            },
        }
    
    # Statistics to get, in order
    foot = {
        'num_obs': {
            'index': 'Num. Observations',
            'value': f'{reg["num_obs"]:0}',
            },
        'rsquared_adj': {
            'index': 'R-squared (adj.)',
            'value': f'{reg["rsquared_adj"]:.6}',
            },
        'aic': {
            'index': 'AIC',
            'value': f'{reg["aic"]:.6}',
            },
        'bic': {
            'index': 'BIC',
            'value': f'{reg["bic"]:.6}',
            },
        'fvalue': {
            'index': 'F-statistic',
            'value': f'{reg["fvalue"]:.6}',
            },
        'f_pvalue': {
            'index': 'F p-value',
            'value': is_fpvalue_small(reg['f_pvalue'])
            }
        }
    
    # Lists for values and indeces
    head_values, head_indexes = [], []
    foot_values, foot_indexes = [], []
    
    # Populate lists based on stats
    for key, prop2val in head.items():
        head_indexes.append(prop2val['index'])
        head_values.append(prop2val['value'])
    for key, prop2val in foot.items():
        foot_indexes.append(prop2val['index'])
        foot_values.append(prop2val['value'])
        
    # Build dataframes for header and footer
    head = pd.DataFrame(head_values, index=head_indexes)
    foot = pd.DataFrame(foot_values, index=foot_indexes)
    
    return head, foot
    
def which_network_measures(include_connected_metrics):
    if include_connected_metrics:
        string = 'All'
    else:
        string = 'Disconn. only'
    return string

def which_task_controls(task_measures_included):
    if task_measures_included == 'both':
        string = 'Meas. & Fixed eff.'
    elif task_measures_included == 'fes':
        string = 'Fixed effects only'
    else: # Only metrics
        string = 'Measures only'
    return string

def is_fpvalue_small(f_pvalue):
    if f_pvalue < 0.001:
        string = '< 0.001'
    else:
        print(f_pvalue)
        string = f'{f_pvalue:.3}'
    return string
    
def star_maker(p):
    if p < 0.001:
        stars = '*'
    else:
        stars = ''
    return stars

def build_multiindex(index):
    ind_list = []
    for ind in index:
        if ind == 'Task controls':
            ind_list.append(('\\textbf{Measure groups}',ind))
        elif ind == 'Intercept':
            ind_list.append(('\\textbf{Controls}',ind))
        elif ind == 'Degree Cent. (Mean)':
            ind_list.append(('\\textbf{Network measures}',ind))
        elif ind == 'Exploration difficulty (1 - Task integral)':
            ind_list.append(('\\textbf{Task measures}',ind))
        elif ind == 'log$_{10}$(Search radius) $\\times$ Degree Cent. (Mean)':
            ind_list.append(('\\textbf{Interactions}',ind))
        elif ind == 'Num. Observations':
            ind_list.append(('\\textbf{Statistics}',ind))
        else:
            ind_list.append(('',ind))
    return ind_list
    
def build_reg_cell(param, stderr, pstar):
    return f'{param}{pstar} ({stderr})'
        
def get_model_names():
    return {0: 'Connected graphs', 1: 'All graphs'}

def build_rf_column(imp, std):
    column = []
    for ii, ss in zip(imp, std):
        column.append(f'{ii:.6f} ({ss:.6f})')
    return column
        
def build_random_forest_tables():

    # Load random forest feature importances and stdevs
    feat = pd.read_pickle('../data/analysis/rf_features.pickle')
    
    # Get names
    metric_labels = up.get_metric_labels()
    other_labels = {
        'run_step': 'Time step',
        'agent_steplim': 'Search radius',
        'team_size': 'Team size'
        }
    
    rftable = []
    
    # Make into table
    for key, model in feat.items():
        rftable.append(pd.DataFrame({
            get_model_names()[key]: build_rf_column(
                model['importances'],
                model['stdevs']
                )},
            index=model['variables']
            ))
    rftable = rftable[0].join(rftable[1],how='outer').fillna('')
        
    # Replace index names
    for key in metric_labels.keys():
        rftable.index \
            = rftable.index.str.replace(key, metric_labels[key], regex=False)
    for key in other_labels.keys():
        rftable.index \
            = rftable.index.str.replace(key, other_labels[key], regex=False)
        
    # Remove fixed effects
    rftable = rftable[~rftable.index.str.contains('C(team_fn)',regex=False)]
    
    print(rftable.style.to_latex(hrules=True))        

if __name__ == '__main__':
    
    netstable = build_network_measures_table()
    # regtable = build_regression_tables()
    # build_random_forest_tables()
