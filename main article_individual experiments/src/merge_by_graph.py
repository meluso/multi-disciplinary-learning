# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:16:46 2021

@author: John Meluso
"""

# Import model functions
import util.analysis as ua
import util.data as ud
import util.variables as uv

All = slice(None)

def merge_by_graph(execset=9, team_size=9, base_graph='empty', melt=False):
    
    # Load data
    loc = f'../data/sets/execset{execset:03}_stats_by_graph.pickle'
    params2stats = ud.load_pickle(loc)
    
    # Subset to performance data
    outcome_vars = list(uv.get_outcomes())
    
    # Get subset with only base stats
    base = params2stats[outcome_vars].xs(
        f'{base_graph}_na_na_na',level='team_graph')
    
    # Rename remaining ones
    base = base.rename(
        columns={
            'count': 'base_count',
            'mean': 'base_mean',
            'std': 'base_std',
            'var': 'base_var'
        })
    
    # Merge with params2stats on non-graph columns
    params2stats = params2stats.reset_index().merge(
        base,
        how='left',
        on=['model_type','team_size','agent_steplim']
        ).set_index(params2stats.index.names)
    
    
    for oc in outcome_vars:
        
        # Calculate confidence interval of mean
        params2stats[oc, 'ci_lo'], params2stats[oc, 'ci_hi'] \
            = ua.conf_int(
                diff_means=params2stats[oc, 'mean'],
                s1=params2stats[oc, 'std'],
                n1=params2stats[oc, 'count'],
                s2=params2stats[oc, 'base_std'],
                n2=params2stats[oc, 'base_count']
                )
        
        # Calculate difference
        params2stats[oc,'diff_mean'] \
            = params2stats[oc,'mean'] - params2stats[oc,'base_mean']
        
        # Calculate confidence interval of diff_mean
        params2stats[oc, 'diff_ci_lo'], params2stats[oc, 'diff_ci_hi'] \
            = ua.conf_int(
                diff_means=params2stats[oc, 'diff_mean'],
                s1=params2stats[oc, 'std'],
                n1=params2stats[oc, 'count'],
                s2=params2stats[oc, 'base_std'],
                n2=params2stats[oc, 'base_count']
                )
        
        # Calculate percent
        params2stats[oc,'pct_mean'] = (params2stats[oc,'mean'] \
           - params2stats[oc,'base_mean'])/params2stats[oc,'base_mean']
        
        # Calculate confidence interval of diff_mean
        params2stats[oc,'pct_ci_lo'] = (params2stats[oc,'ci_lo'] \
           - params2stats[oc,'base_mean'])/params2stats[oc,'base_mean']
        params2stats[oc,'pct_ci_hi'] = (params2stats[oc,'ci_hi'] \
           - params2stats[oc,'base_mean'])/params2stats[oc,'base_mean']
            
    # Melt down to graphable form
    keep_vars = outcome_vars.copy()
    keep_vars.append('team_graph_density')
    keep_vars.append('team_graph_centrality_eigenvector_mean')
    cols = (keep_vars, ['mean','ci_lo','ci_hi','diff_mean','diff_ci_lo',
                        'diff_ci_hi','pct_mean','pct_ci_lo','pct_ci_hi'])
    output = params2stats.xs(team_size,level='team_size').loc[:,cols]
        
    # Pickle stats
    loc = f'../data/sets/execset{execset:03}_by_graph_vs_{base_graph}.pickle'
    output.to_pickle(loc)
    
    return output
    
if __name__ == '__main__':
    df = merge_by_graph()