# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 13:50:34 2022

@author: John Meluso
"""

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd

# Import source files
import fig_settings as fs
import util.plots as up

### Loading functions

def load_one_regression(reg):
        
    # Load slim results for each regression
    fname = f'../data/analysis/reg_slim_{reg:02}.pickle'
    results = pd.read_pickle(fname)
    
    # Create list of stats
    stats = ['params','HC2_se','pvalues']
    
    # Combine stats into a dataframe
    results['df'] = pd.concat(
        [results['params'], results['HC2_se'], results['pvalues']],
        axis=1
        )
    results['df'].columns = ['params','HC2_se','pvalues']
    
    for stat in stats:
        
        # Remove dataframed stats
        results.pop(stat)
        
        # Remove insignificant values
        results['df'][stat].loc[results['df']['pvalues']>0.001] = 0
    
    # Build masks for data selection
    mask = {}
    mask['ntwk'] = results['df'].index.str.contains(
        'team_graph',regex=False)
    mask['task'] = results['df'].index.str.contains(
        'team_fn_',regex=False)
    mask['main'] = ~results['df'].index.str.contains(
        'log10(agent_steplim)',regex=False)
    mask['intx'] = results['df'].index.str.contains(
        'log10(agent_steplim):',regex=False)
    
    # Get main and interaction effects for each group
    for group in ['ntwk','task']:
        
        # Define variable names
        main = f'{group}_main'
        intx = f'{group}_intx'
    
        # Get main and interaction data for networks
        results[main] = results['df'].loc[(mask[group] & mask['main']), :]
        results[intx] = results['df'].loc[(mask[group] & mask['intx']), :]
        
        # Update the interaction effect indeces
        results[intx].set_index(results[intx].index.str.replace(
            'log10(agent_steplim):','',regex=False),
            inplace=True
            )
        
        # Sort effects
        effect_sum = results[main]['params'] + results[intx]['params']
        effect_sum = effect_sum.sort_values(ascending=False)
        sort_order = effect_sum.sort_values(ascending=False).index
        results[main] = results[main].reindex(sort_order)
        results[intx] = results[intx].reindex(sort_order)
        
        # Build interaction anchor
        results[intx]['anchor'] = results[main]['params']
        prod = results[main]['params']*results[intx]['params']
        results[intx]['anchor'][prod < 0] = 0
    
    return results

def get_direct_likes(group2prop2val):
    
    # Declare regressions to include and steps to include
    regs = [3,4,5,9,10,11]
    # regs = [3,5,9,11]
    logsteps = [-2, -1, 0]
    
    # Load regression results
    reg2results = {ii: [] for ii in regs}
    for reg in reg2results.keys():
        reg2results[reg] = load_one_regression(reg)
        
    # Create group to variable name converter
    var2group = {
        'task': 'team_fn_',
        'ntwk': 'team_graph',
        }
    
    # Create variable name to counter converter
    var2counter = {
        var: pd.Series(0, index=reg2results[5][f'{var}_main'].index)
        for var in var2group.keys()
        }
    
    # Create variable name totals (for later division)
    totals = var2counter.copy()
        
    # Count number of effects in each direction
    for var in var2counter.keys():
        for reg in reg2results.keys():
            for ls in logsteps:
            
                # Set var names
                main = f'{var}_main'
                intx = f'{var}_intx'
                
                # If the measure is statistically significant, count it.
                totals[var] = totals[var].add(
                    reg2results[reg][main]['params'] \
                        + ls*reg2results[reg][intx]['params'] != 0,
                    fill_value=0
                    )
                
                # Is task main + intx greater than 0?
                var2counter[var] = var2counter[var].add(
                    reg2results[reg][main]['params'] \
                        + ls*reg2results[reg][intx]['params'] > 0,
                    fill_value=0
                    )
        
        # Normalize counters to [0,1] based on 2 x number regressions
        # (2 from one w/ & w/o interactions)
        group2prop2val[var2group[var]]['like'] \
            = var2counter[var].divide(totals[var])
            
        # Replace any instances of logs
        group2prop2val[var2group[var]]['like'].index \
            = group2prop2val[var2group[var]]['like'].index.str.replace(
                'log10(','',regex=False)
        group2prop2val[var2group[var]]['like'].index \
            = group2prop2val[var2group[var]]['like'].index.str.replace(
                ')','',regex=False)
    
    return group2prop2val

def load_importances():
    
    # Load random forest feature importances and stdevs
    feat = pd.read_pickle('../data/analysis/rf_features.pickle')
    
    # Get names
    model_names = get_model_names()
    
    # Make into table
    table = []
    for key, model in feat.items():
        df = pd.DataFrame(model)
        df['model'] = model_names[key]
        table.append(df)
    df = pd.concat(table)
        
    # Define groups for two axes
    group2prop2val = {
        'team_fn_': {
            'title': 'Task measure importances',
            'meas_types': ['task']
            },
        'team_graph': {
            'title': 'Network measure importances',
            'meas_types': ['conn_to_conn','efficiency','grouping','conn']
            },
        }
    
    # Subset data for each group
    for group in group2prop2val.keys():
        
        # Build dataframe
        group2prop2val[group]['df'] = df[df.variables.str.contains(group)]
        
        # Get pivot tables of importances
        importances = group2prop2val[group]['df'].pivot(
            index='variables',
            columns='model',
            values='importances'
            ).sort_index(axis=1, ascending=False)
        
        # Get pivot table of standard deviations
        stdevs = group2prop2val[group]['df'].pivot(
            index='variables',
            columns='model',
            values='stdevs'
            ).sort_index(axis=1, ascending=False)
        
        # Calculate average importances
        group2prop2val[group]['imp'] = importances.mean(axis=1)
        
        # Calculate average standard deviations
        group2prop2val[group]['std'] = mean_std(stdevs)
        
        # Get sorting order
        order = pd.Categorical(
            group2prop2val[group]['imp'].index,
            categories = group2prop2val[group]['imp'].sort_values().index
            )
            
        # Sort imps and stds by importances
        group2prop2val[group]['imp'] \
            = sort_df_by_order(group2prop2val[group]['imp'], order)
        group2prop2val[group]['std'] \
            = sort_df_by_order(group2prop2val[group]['std'], order)
        
        # Get widths for plotting
        group2prop2val[group]['width'] = len(group2prop2val[group]['imp'])
        
    return group2prop2val

def load_metrics_groups():
    
    # Load metric labels
    metric2prop2val = up.get_metrics()

    # Load random forest importances
    group2prop2val = load_importances()
    
    # Get directional likelihoods from regressions
    group2prop2val = get_direct_likes(group2prop2val)
    
    # Add formatting 
    meas2prop2val, group2prop2val \
        = get_formatting(group2prop2val, metric2prop2val)
    
    # Build new dataframe for imps, stds, likes, and colors
    group2prop2val = build_dataframe(group2prop2val)
    
    # Add group labels
    group2prop2val = get_group_labels(group2prop2val)
    
    return metric2prop2val, group2prop2val, meas2prop2val

### Utility functions

def mean_std(df):
    return ((df**2).sum(axis=1).divide(df.count(axis=1)))**(1/2)
    
def sort_df_by_order(df, order):
    df = df.reset_index()
    df['variables'] = pd.Categorical(df['variables'], categories=order)
    df = df.sort_values('variables', ascending=False).set_index('variables')
    return df

def get_model_names():
    return {0: 'Connected graphs', 1: 'All graphs'}

def get_group_labels(group2prop2val):
    group2prop2val['team_fn_']['label'] = '(a) task measures'
    group2prop2val['team_graph']['label'] = '(b) network measures'
    return group2prop2val

def build_dataframe(group2prop2val):
    for group in group2prop2val.keys():
        group2prop2val[group]['df'] = pd.concat([
            group2prop2val[group]['imp'],
            group2prop2val[group]['std'],
            group2prop2val[group]['like'],
            group2prop2val[group]['color'],
            group2prop2val[group]['marker'],
        ],axis=1)
        group2prop2val[group]['df'].columns \
            = ['imp', 'std', 'like', 'color', 'marker']
    return group2prop2val

def get_formatting(group2prop2val, metric2prop2val):
    meas2prop2val = up.get_measure_formatting()
    for group in group2prop2val.keys():
        for prop in ['color', 'marker']:
            
            # Define lambda functions
            meas_lambda = lambda x: metric2prop2val[x]['meas_type']
            prop_lambda = lambda x: meas2prop2val[x][prop]
            
            # Values to look up
            metrics = group2prop2val[group]['imp'].index.to_series()
            
            # Where to look them up, first get meas_type
            meas_types = metrics.apply(meas_lambda)
            
            # Then use meas_type to get each prop
            group2prop2val[group][prop] = meas_types.apply(prop_lambda)
            
    return meas2prop2val, group2prop2val

def arrow(ax, xyfrom, xyto, text=''):
    ax.annotate(text=text, xy=xyto, xytext=xyfrom, annotation_clip=False,
        arrowprops=dict(arrowstyle='->',fc='#AAAAAA',ec='#AAAAAA'),
        xycoords='axes fraction')
    
def draw_lollipop(ax, x, y, error, marker, color, size, label):
    
    y_anchor = 0.5
    if y > y_anchor:
        ymin, ymax = y_anchor, y
    else:
        ymin, ymax = y, y_anchor
    ax.vlines(x=x, ymin=ymin, ymax=ymax, colors='#BBBBBB', linewidths=1, zorder=0)
    ax.scatter(
        x=x,
        y=y,
        s=size,
        marker=marker,
        facecolors=color,
        edgecolors='#FFFFFF',
        linewidths=0.5,
        label=label
        )
    
def add_label(ax, text, xy, xytext, ha, va, use_connector=False):
    
    if use_connector:
        arrowprops=dict(
            arrowstyle='-',
            lw=0.3,
            ls='-',
            edgecolor='#222222',
            shrinkA=0
            )
    else:
        arrowprops=None
        
    if xytext is not None:
        textcoords='offset points'
    else:
        textcoords=None
    
    ax.annotate(
        text=text,
        xy=xy,
        xytext=xytext,
        textcoords=textcoords,
        annotation_clip=False,
        size=6,
        color='#222222',
        ha=ha,
        va=va,
        arrowprops=arrowprops
        )
    
def build_list_iterator(prop2val):
    
    metrics = prop2val['df'].index
    xlist = prop2val['df']['imp']
    ylist = prop2val['df']['like']
    elist = prop2val['df']['std']
    mlist = prop2val['df']['marker']
    clist = prop2val['df']['color']
    
    return zip(metrics, xlist, ylist, elist, mlist, clist)

### Plotting functions
    
def plot_importances(group, group2prop2val, metric2prop2val, meas2prop2val,
                     fig, axs):
    
    # Get prop2val
    prop2val = group2prop2val[group]
    
    # Plot (x) importances vs (y) likelihoods
    for aa, (ax, mtype) in enumerate(zip(axs, group2prop2val[group]['meas_types'])):
        
        # Plot each point individually
        iterator = build_list_iterator(prop2val)
        label = prop2val['label']
        
        # Loop through points and plot individually
        for metric, x, y, er, m, c in iterator:
            
            # Add color and label if we're in the measure's group
            if metric2prop2val[metric]['meas_type'] == mtype:
                color = c
                size = 40
                add_label(
                    ax=ax,
                    text=metric2prop2val[metric]['label'],
                    xy=(x,y),
                    xytext=metric2prop2val[metric]['annot_loc'],
                    ha=metric2prop2val[metric]['ha'],
                    va=metric2prop2val[metric]['va'],
                    use_connector=metric2prop2val[metric]['use_conn']
                    )
            
            # Otherwise, just make it gray
            else:
                color = '#AAAAAA'
                size = 30
            
            # Create lollipop at point
            draw_lollipop(ax, x, y, er, m, color, size, label)
        
        # Set y ticks
        ticks = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
        labels = [
            'Likely negative\n(100% neg. effects)',
            '','','',
            'Evenly\npos. & neg.',
            '','','',
            'Likely positive\n(100% pos. effects)'
             ]
        ax.set_yticks(ticks, labels, va='center')
        
        # Add labels and lines
        fs.set_border(ax, left=True, bottom=True)
        ax.spines['bottom'].set_position(('data',0.5))
        ax.set_axisbelow(True)
        ax.set_title(meas2prop2val[mtype]['label'])
        
        # Set x ticks
        if aa == 0 or aa == len(axs) - 1:
            ax.tick_params(axis='x', pad=57)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        
    return fig, axs
