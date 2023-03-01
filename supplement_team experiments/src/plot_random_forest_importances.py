# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:47:05 2022

@author: John Meluso
"""

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd

# Import source files
import fig_settings as fs
import util.plots as up

fs.set_fonts()

def plot_randfor_imps(save=True):

    # Load random forest feature importances and stdevs
    feat = pd.read_pickle('../data/analysis/rf_features.pickle')
    
    # Get names
    metric_labels = up.get_metric_labels()
    model_names = get_model_names()
    
    # Make into table
    table = []
    for key, model in feat.items():
        df = pd.DataFrame(model)
        df['model'] = model_names[key]
        table.append(df)
        
    # Define groups for two axes
    groups = {
        'team_fn_': 'Task measure importances',
        'team_graph': 'Network measure importances',
        }
    
    # Subset data for each group
    df = pd.concat(table)
    data = {group: df[df.variables.str.contains(group)] for group in groups}
        
    
    # Plot the features
    fig, axs = plt.subplots(1, len(groups),
        figsize=fs.fig_size(0.8, 0.45),
        sharey=True,
        dpi=1200,
        gridspec_kw=dict(
            width_ratios = [len(values) for values in data.values()]
            )
        )
    
    # Sort by connected feature imporance
    for ii, ((group, title), ax) in enumerate(zip(groups.items(), axs)):
    
        # Get pivot tables of importances and stdevs
        importances = data[group].pivot(
            index='variables',
            columns='model',
            values='importances'
            ).sort_index(axis=1, ascending=False)
        stdevs = data[group].pivot(
            index='variables',
            columns='model',
            values='stdevs'
            ).sort_index(axis=1, ascending=False)
    
        # Get sorting order
        means = importances.mean(axis=1)
        order = pd.Categorical(
            means.index,
            categories = importances.mean(axis=1).sort_values().index
            )
        
        # Sort importance means, and stdevs by order
        importances = sort_df_by_order(importances, order)
        means = sort_df_by_order(means, order)
        stdevs = sort_df_by_order(stdevs, order)
        
        means.plot(kind='line', ax=ax, color='#222222', style='.-', legend=False)
        importances.plot(kind='bar', ax=ax, yerr=stdevs, width=0.8,
                         legend=False, xlabel='', color=('tab:blue','tab:orange'))
        # ax.set_title('Random forest feature importances')
        ax.set_ylabel('Feature importance')
        ax.set_xticks(
            ticks=range(len(importances.index)),
            labels=[metric_labels[metric] for metric in importances.index],
            ha='right',
            rotation=45,
            rotation_mode='anchor'
            )
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(title)
        ax.set_ylim(ymin=0.0003)
        
        
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[1], handles[2], handles[0]]
    labels = [
        labels[1] + ' (obs.$=5.5345E+07$)',
        labels[2] + ' (obs.$=6.3648E+07$)',
        'Mean feature importance'
        ]
    fig.legend(
        handles,
        labels,
        # bbox_to_anchor=(0.5, 0),
        loc='lower center',
        frameon=False,
        ncol=3
        )
    fig.tight_layout(rect=(0,0.035,1,1), w_pad=-3)
    if save: fs.save_pub_fig('random_forest_feature_importances')
    
def sort_df_by_order(df, order):
    df = df.reset_index()
    df['variables'] = pd.Categorical(df['variables'], categories=order)
    df = df.sort_values('variables', ascending=False).set_index('variables')
    return df

def get_model_names():
    return {0: 'Connected graphs', 1: 'All graphs'}

if __name__ == '__main__':
    plot_randfor_imps()
