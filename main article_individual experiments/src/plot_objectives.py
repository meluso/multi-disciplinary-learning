# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 09:23:46 2022

@author: John Meluso
"""

# Import libraries
import itertools as it
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from string import ascii_lowercase

# Import src classes
import classes.Objective as Objective
import util.functions as uf
import util.writing as uw
import fig_settings as fs

fs.set_fonts()

def iter_all_strings():
    for size in it.count(1):
        for s in it.product(ascii_lowercase, repeat=size):
            yield "".join(s)

def test_objective(fn_type, fn_opts):
    '''Tests the objective specified by name.'''
 
    # Get fn opts
    opts = uf.get_fn_opts(fn_type, fn_opts)
    
    # Set some constants
    x_min = 0
    x_max = 1
    ndivs = 100
    ks = [1,5]
    
    # Create adjusted colormap
    cmap = sns.color_palette('rocket', as_cmap=True)
    
    # Set x and y ranges
    x_range = np.linspace(x_min,x_max,ndivs)
    y_range = np.linspace(x_min,x_max,ndivs)
    x_mesh, y_mesh = np.meshgrid(x_range,y_range,indexing='ij')
    z_mesh = np.round(np.zeros((ndivs,ndivs)),decimals=3)
    
    # Create objective
    objective = get_objective(fn_type, opts, ks)
    
    # Update zmesh
    for (ii,xx), (jj,yy) \
        in it.product(enumerate(x_range),enumerate(y_range)):
        
        z_mesh[ii,jj] = objective([xx,yy])
        
    # Plot the zmesh
    fig = plt.figure(dpi=1200, constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=25,azim=-35)
    surf = ax.plot_surface(x_mesh,y_mesh,z_mesh,cmap=cmap)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$g(x_1,x_2)$')
    
    # Build print string
    file_name = fn_type
    keys = ['exponent','frequency','weight']
    for key in keys:
        if key in opts.keys():
            file_name += f'_{opts[key]}'
        else:
            file_name += '_na'
    
    return fig, file_name

def objective_subfig(subfig, fn_type, fn_opts, fn_letter):
    '''Plots the objective specified by name on the subfig.'''
 
    # Get fn opts
    opts = uf.get_fn_opts(fn_type, fn_opts)
    
    # Set some constants
    x_min = 0
    x_max = 1
    ndivs = 100
    ks = [1,5]
    
    # Create adjusted colormap
    cmap = sns.color_palette('rocket', as_cmap=True)
    
    # Set x and y ranges
    x_range = np.linspace(x_min,x_max,ndivs)
    y_range = np.linspace(x_min,x_max,ndivs)
    x_mesh, y_mesh = np.meshgrid(x_range,y_range,indexing='ij')
    z_mesh = np.round(np.zeros((ndivs,ndivs)),decimals=3)
    
    # Create objective
    objective = get_objective(fn_type, opts, ks)
    
    # Update zmesh
    for (ii,xx), (jj,yy) \
        in it.product(enumerate(x_range),enumerate(y_range)):
        
        z_mesh[ii,jj] = objective([xx,yy])
        
    # Plot the zmesh
    ax = subfig.add_subplot(111, projection='3d')
    #ax.view_init(elev=25,azim=-35)
    surf = ax.plot_surface(x_mesh,y_mesh,z_mesh,cmap=cmap)
    ax.tick_params(pad=-2)
    ax.set_xlabel('$x_1$', labelpad=-5)
    ax.set_ylabel('$x_2$', labelpad=-5)
    ax.text(x=0.8, y=1, z=1.2, s='$g(x_1,x_2)$')
    ax.margins(tight=True)
    
    # Build print string
    file_name = fn_type
    keys = ['exponent','frequency','weight']
    for key in keys:
        if key in opts.keys():
            file_name += f'_{opts[key]}'
        else:
            file_name += '_na'
            
    # Build title text
    name = uw.get_fns()[file_name]['name']
    title = f'({fn_letter}) {name}'
    title = title.replace(' (','\n(')
    title = title.replace('\sinsq','Sin$^2$')
    title = title.replace('\\texorpdfstring{$K+1$}{K+1}','K$+1$')
    
    # Add title
    ax.set_title(title, pad=-15)
    
    return subfig, file_name

def gen_tex(text, fn_type, page, letter):
    
    props = uw.get_fns()[fn_type]
    
    if 'extra' in props.keys():
        extra = props['extra'] + ' '
    else:
        extra = ''
    
    text += f'\subsection{{{props["name"]}}}\n'
    text += f'\label{{task:{fn_type}}}\n'
    text +=  f'{extra}See Fig.~\\ref{{fig:tasks{page}}}{letter} on ' \
        + f'page~\\pageref{{fig:tasks{page}}} ' \
        + 'for a 2-variable example where $k_1=1$ and $k_2=5$.\n'
    text += '\\begin{equation}\n',
    text += f'\ty(\Vec{{x}}_m)={props["equation"]}\n',
    text += '\end{equation}'
    text += '\n\n'
    
    return text
    

def plot_all_objectives():
    '''Tests all of the objectives.'''
    
    all_functions = uf.set_all_functions_default()
    
    # Loop through all functions
    for fn, fn_opts in all_functions.items():
        
        # Plot each combination of function options
        for opts in list(it.product(*fn_opts.values())):
            
            # Create the plot
            fig, fn_type = test_objective(fn,opts)
            
            # Save the plot to file
            for ff in ['eps','png']:
                loc = f'../figures/objectives/publication/{ff}/'
                name = fn_type + f'.{ff}'
                file = loc + name
                plt.savefig(file, format=ff, dpi=1200)
                
def plot_objective_pages():
    '''Tests all of the objectives.'''
    
    all_functions = uf.set_all_functions_default()
    task_list = []
    plot_dict = {}
    tasks_per_fig = 12
    text = []
    
    # Build list of functions and options
    for fn, fn_opts in all_functions.items():
        for opts in list(it.product(*fn_opts.values())):
            task_list.append({
                'fn_type': fn,
                'fn_opts': opts
                })
            
    # Get letter prefix for each title
    for ii, s in enumerate(it.islice(iter_all_strings(), len(task_list))):
        task_list[ii]['letter'] = s
            
    # Create pages, each with up to 20 tasks
    for page in range(len(task_list)//tasks_per_fig + 1):
    
        # Create figure and subfigures
        fig = plt.figure(
            figsize=fs.fig_size(1, 0.9),
            dpi=1200,
            layout='constrained',
            )
        subfigs = fig.subfigures(nrows=4, ncols=3, wspace=0.1, hspace=0.1)
    
        # Add plots to figure and create latex file for it, too
        for ii, subfig in enumerate(subfigs.flat):
            curr_plot = page*tasks_per_fig + ii
            if curr_plot < len(task_list):
                subfig, fn_type = objective_subfig(
                    subfig=subfig,
                    fn_type=task_list[curr_plot]['fn_type'],
                    fn_opts=task_list[curr_plot]['fn_opts'],
                    fn_letter=task_list[curr_plot]['letter']
                    )
                text = gen_tex(
                    text,
                    fn_type,
                    page,
                    task_list[curr_plot]['letter']
                    )
            
        # Save the page to file
        fs.save_pub_fig(f'tasks{page}')
    
    # Write text to file
    with open('../tex/tasks.tex','w') as file:
        file.writelines(text)
    
        
def get_objective(fn_type,fn_opts,degrees):
    '''Selects and returns a callable object of the correct function type.'''
    
    try:
        return getattr(Objective,fn_type.capitalize())(fn_opts,degrees)
    except:
        raise RuntimeError('Function type ' + fn_type + ' is not valid.')


if __name__ == '__main__':
    
    plot_objective_pages()