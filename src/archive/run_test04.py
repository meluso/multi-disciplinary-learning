# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:41:53 2021

@author: John Meluso
"""
import itertools as it
import matplotlib.pyplot as plt
import model_team as mt
import pandas as pd
import seaborn as sns

if __name__ == '__main__':
    
    # Test team
    test = 4
    graphs = ['empty','complete','small_world','random','power']
    sizes = [10,25,100]
    fn_classes = ['root','average','sphere','sin2','sin2sphere',
                  'posexpdeg','negexpdeg']
    fn_subclasses = ['equalwt','degreewt']
    steps = 5
    runs = 5
    suffixes = ['unwt','wt']
    points = {'graph': [],'fn_class': [],'fn_subclass': [],'nodes': [],
              'run': [],'step': [],'average': [],'avg_type': []}
    
    # Create a team
    for graph, nodes, fncl, fnsb, run in list(
            it.product(graphs, sizes, fn_classes, fn_subclasses, range(runs))
            ):
        team = mt.Team(graph,nodes,fncl,fnsb)
        for step in range(steps):
            team.step()
            for suffix in suffixes:
                points['graph'].append(graph)
                points['fn_class'].append(fncl)
                points['fn_subclass'].append(fnsb)
                points['nodes'].append(nodes)
                points['run'].append(run)
                points['step'].append(step)
                points['average'].append(team.get_fxs_avg(suffix))
                points['avg_type'].append(suffix)
    
    # Convert points to data frame
    df = pd.DataFrame(points)
    
    #%% Construct baseline from empty graph
    df_empty = df.loc[df.graph == 'empty']
    df_empty = df_empty.drop(columns=['graph','run'])
    df_empty = df_empty.groupby(by=['fn_class','fn_subclass','nodes','step'])
    df_empty_mean = df_empty.mean()
    
    
    #%% Plot results
    fig = plt.figure(figsize=(25,14),dpi=300)
    axs = fig.subplots(len(sizes),len(fn_classes))
    
    for row, nodes in enumerate(sizes):
        for col, fncl in enumerate(fn_classes):
            data = df.loc[(df.fn_class == fncl) & (df.nodes == nodes)]
    
            sns.set_theme()
            sns.lineplot(x='step',y='average',hue='graph',style='fn_subclass',
                         data=data, ax=axs[row,col])
            axs[row,col].set_title('nodes: '+str(nodes)+', fn: '+fncl)
            
    fig.tight_layout()
    plt.savefig('../../figures/run_test_' + str(test) + '.png',dpi=300,
                bbox_inches='tight')