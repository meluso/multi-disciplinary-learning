# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 13:50:34 2022

@author: John Meluso
"""

# Import libraries
import matplotlib.pyplot as plt

# Import source files
import fig_settings as fs
import plot_importances as pi
import util.plots as up

fs.set_fonts({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'cm',
    'legend.fontsize': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'figure.titlesize': 6
    })
    
def plot_importances_networks(save=True):
    
    # Load metric and group dicts
    metric2prop2val, group2prop2val, meas2prop2val = pi.load_metrics_groups()
    
    # Set group
    group = 'team_graph'
        
    # Construct a figure, with one ax for each network measure group
    fig, axs = plt.subplots(4, 1, figsize=fs.fig_size(0.5, 0.85, 2),
                            dpi=1200,
                            layout='constrained',
                            gridspec_kw={'hspace': 0.1}
                            )
    
    # Set mins and maxes
    xmin, xmax, ymin, ymax = 0, 0.003, -0.05, 1.05
    ximp = 0.25
    
    # Get prop2val
    prop2val = group2prop2val[group]
    
    # Plot (x) importances vs (y) likelihoods
    for aa, (ax, mtype) in enumerate(zip(axs, group2prop2val[group]['meas_types'])):
        
        # Plot each point individually
        iterator = pi.build_list_iterator(prop2val)
        label = prop2val['label']
        
        # Loop through points and plot individually
        for metric, x, y, er, m, c in iterator:
            
            # Add color and label if we're in the measure's group
            if metric2prop2val[metric]['meas_type'] == mtype:
                color = c
                size = 36
                pi.add_label(
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
            pi.draw_lollipop(ax, x, y, er, m, color, size, label)
        
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
        
    # Make adjustments to each all axes
    for ax in axs:
        
        # Set x limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
    # Create x label
    axs[0].set_xlabel('Measure importance')
    axs[-1].set_xlabel('Measure importance')
        
    # Create y label
    axs[0].text(x=-0.05, y=1.15, s='Effect direction\n& likelihood',
                transform=axs[0].transAxes, size=8, ha='center', va='center')
    
    # Guiding arrows & text
    up.arrow(axs[0], (-0.04, 0.35), (-0.04, 0.15))
    up.arrow(axs[0], (-0.04, 0.65), (-0.04, 0.85))
    up.arrow(axs[0], (ximp-0.075, 0.54), (ximp+0.075, 0.54))
    axs[0].text(x=-0.07, y=0.25, s='More\nlikely\nnegative',
                size=6, clip_on=False, va='center', ha='right', color='#666666',
                transform=axs[0].transAxes)
    axs[0].text(x=-0.07, y=0.75, s='More\nlikely\npositive',
                size=6, clip_on=False, va='center', ha='right', color='#666666',
                transform=axs[0].transAxes)
    axs[0].text(x=ximp, y=0.57, s='More important', size=6, 
                clip_on=False, va='bottom', ha='center', color='#666666',
                transform=axs[0].transAxes)
    
    print(axs[0].bbox.height)
    if save: fs.save_pub_fig('directional_importances_networks', bbox_inches='tight')

def plot_importances_by_group(save=True):
    
    # Load metric and group dicts
    metric2prop2val, group2prop2val, meas2prop2val = pi.load_metrics_groups()
    
    # Set group
    group = 'team_graph'
    
    # Set mins and maxes
    xmin, xmax, ymin, ymax = 0, 0.003, -0.05, 1.05
    ximp = 0.25
    
    # Get prop2val
    prop2val = group2prop2val[group]
    
    # Add empty axes and none groups
    new_groups = ['none','empty']
    for var in new_groups:
        prop2val['meas_types'].append(var)
        
    # Plot (x) importances vs (y) likelihoods
    for mtype in group2prop2val[group]['meas_types']:
        
        # Construct a figure for the network measure group
        fig = plt.figure(figsize=fs.fig_size(0.5, 0.2, 2), dpi=1200)
        ax = fig.gca()
        
        # Plot each point individually
        iterator = pi.build_list_iterator(prop2val)
        label = prop2val['label']
        
        # Loop through points and plot individually
        for metric, x, y, er, m, c in iterator:
            
            # Add color and label if we're in the measure's group
            if metric2prop2val[metric]['meas_type'] == mtype:
                color = c
                size = 36
                pi.add_label(
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
            if mtype != 'empty':
                pi.draw_lollipop(ax, x, y, er, m, color, size, label)
                
        
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
        if mtype not in new_groups:
            ax.set_title(meas2prop2val[mtype]['pres-label'])
        
        # Set x ticks
        ax.tick_params(axis='x', pad=57)
        
        # Set x limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
            
        # Create x label
        ax.set_xlabel('Measure importance')
        ax.set_xlabel('Measure importance')
            
        # Create y label
        ax.text(x=-0.05, y=1.15, s='Effect direction\n& likelihood',
                transform=ax.transAxes, size=8, ha='center', va='center')
        
        # Guiding arrows & text
        up.arrow(ax, (-0.04, 0.35), (-0.04, 0.15))
        up.arrow(ax, (-0.04, 0.65), (-0.04, 0.85))
        up.arrow(ax, (ximp-0.075, 0.54), (ximp+0.075, 0.54))
        ax.text(x=-0.07, y=0.25, s='More\nlikely\nnegative',
                    size=6, clip_on=False, va='center', ha='right', color='#666666',
                    transform=ax.transAxes)
        ax.text(x=-0.07, y=0.75, s='More\nlikely\npositive',
                    size=6, clip_on=False, va='center', ha='right', color='#666666',
                    transform=ax.transAxes)
        ax.text(x=ximp, y=0.57, s='More important', size=6, 
                    clip_on=False, va='bottom', ha='center', color='#666666',
                    transform=ax.transAxes)
        
        fname = f'directional_importances_networks_{mtype}'
        if save:
            fs.save_presentation_fig(fname, bbox_inches='tight')
        else:
            print(fname)

if __name__ == '__main__':
    # plot_importances_networks()
    plot_importances_by_group()
