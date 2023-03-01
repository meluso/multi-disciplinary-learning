# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:51:31 2022

@author: jam
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import source files
import fig_settings as fs

fs.set_fonts()

def load_slim_results(filename):
    return pd.read_pickle(filename)

def get_labels(connected=True):
    
    if connected:
        fn2label = {
            'team_graph_centrality_degree_mean': 'Degree Cent. (Mean)',
            'team_graph_centrality_degree_stdev': 'Degree Cent. (St. Dev.)',
            'team_graph_centrality_eigenvector_mean': 'Eigenvector Cent. (Mean)',
            'team_graph_centrality_eigenvector_stdev': 'Eigenvector Cent. (St. Dev.)',
            'team_graph_centrality_betweenness_mean': 'Betweenness Cent. (Mean)',
            'team_graph_centrality_betweenness_stdev': 'Betweenness Cent. (St. Dev.)',
            'team_graph_nearest_neighbor_degree_mean': 'Nearest Neighbor Degree (Mean)',
            'team_graph_nearest_neighbor_degree_stdev': 'Nearest Neighbor (St. Dev.)',
            'team_graph_clustering': 'Clustering Coeff.',
            'team_graph_assortativity': 'Assortativity',
            'team_graph_pathlength': 'Shortest Pathlength',
            'team_graph_diameter': 'Diameter'
            }
    else:
        fn2label = {
            'team_graph_centrality_degree_mean': 'Degree Cent. (Mean)',
            'team_graph_centrality_degree_stdev': 'Degree Cent. (St. Dev.)',
            'team_graph_centrality_eigenvector_mean': 'Eigenvector Cent. (Mean)',
            'team_graph_centrality_eigenvector_stdev': 'Eigenvector Cent. (St. Dev.)',
            'team_graph_centrality_betweenness_mean': 'Betweenness Cent. (Mean)',
            'team_graph_centrality_betweenness_stdev': 'Betweenness Cent. (St. Dev.)',
            'team_graph_nearest_neighbor_degree_mean': 'Nearest Neighbor Degree (Mean)',
            'team_graph_nearest_neighbor_degree_stdev': 'Nearest Neighbor (St. Dev.)',
            'team_graph_clustering': 'Clustering Coeff.'
            }

    mod2label2val = {
        '3xx': {
            'title': 'Model 1',
            'indep_label': 'Network Metrics'
            },
        '3xg': {
            'title': 'Model 2',
            'indep_label': 'Network Metrics'
            }
        }
    
    out2label2val = {
        'team_performance': {
            'title': 'Team Performance',
            'dep_label': 'Regression Coefficient (linear)'
            },
        'team_productivity': {
            'title': 'Agent Productivity',
            'dep_label': 'Regression Coefficient ($log_{10}$)'
            },
        }
    
    return mod2label2val, out2label2val, fn2label
    
def generate_figure_column(connected=True):
    
    # Load model2outcome2property2value
    if connected:
        filename = '../data/regression/regression_by_subfns_connected_slim.pickle'
    else:
        filename = '../data/regression/regression_by_subfns_all_slim.pickle'
    mod2out = load_slim_results(filename)
    
    # get labels
    mod2label2val, out2label2val, fn2label = get_labels(connected)
    show_vars = fn2label.keys()
    
    fig, axs = plt.subplots(nrows=len(mod2label2val),ncols=1,
                            figsize=(6,9), sharex='col', sharey='row')    
    
    for oo, outcome in enumerate(out2label2val.keys()):
        for mm, model in enumerate(mod2label2val.keys()):
            
            # Get data subset
            desc2val = mod2out[model][outcome]
            df = pd.concat((
                desc2val['params'].rename('params'),
                desc2val['HC2_se'].rename('HC2_se'),
                desc2val['pvalues'].rename('pvalues')
                ), axis=1)
            df = df.loc[show_vars,:].sort_values(by='params', ascending=False)
            
            # Get data to plot
            indep_var = df.index
            dep_var = df.params
            dep_err = df.HC2_se
            dep_sig = df.pvalues
            
            # Label locations and set bar width
            x = np.arange(len(indep_var))  # the label locations
            width = 0.35  # the width of the bars
            
            # Build legend entry
            model_str = mod2label2val[model]['title']
            rsq = desc2val['rsquared_adj']
            # pval = desc2val['f_pvalue']
            adjR2_str = f'Adj. $R^2={rsq:.3f}$'
            pval_str = 'p-value$<0.001$'
            leg_ent = f'{model_str} ({adjR2_str}, {pval_str})'
            
            # Plot values
            x_locs = x - width/2 + mm*width
            bars = axs[oo].bar(x=x_locs, height=dep_var, yerr=dep_err,
                               width=width, capsize=3, label=leg_ent)
            
            # Add p-value labels to bars
            p_labels = pd.Series(['*' if xx <= 0.001 else '' for xx in dep_sig])
            axs[oo].bar_label(bars, labels=p_labels, padding=3)
            
            # Add axis labels
            if mm==0: axs[oo].set_ylabel(out2label2val[outcome]['dep_label'])
            if oo==1:
                axs[oo].set_xlabel(mod2label2val[model]['indep_label'])
                
        # Add tick labels
        axs[oo].set_xticks(
            ticks=x,
            labels=[fn2label[key] for key in indep_var],
            rotation=45,
            ha='right'
            )
        
        # Add grid and bold axis
        axs[oo].set_title(out2label2val[outcome]['title'])
        axs[oo].legend()
        axs[oo].grid(True)
        axs[oo].axhline(linewidth=2,color='#0F0F0F')
        axs[oo].set_axisbelow(True)
        axs[oo].margins(y=0.1)
        
    # Create figure name
    if connected:
        name = 'regression_coefficients_connected_graphs'
    else:
        name = 'regression_coefficients_all_graphs'
    
    
    # Show
    plt.tight_layout()
    fs.save_pub_fig(name)
    plt.show()
    

if __name__ == '__main__':
    generate_figure_column(connected=True)
    generate_figure_column(connected=False)