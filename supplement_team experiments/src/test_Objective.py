# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:04:58 2021

@author: John Meluso
"""

# Import libraries
import itertools as it
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

# Import src classes
import classes.Objective as Objective
import util.functions as uf


def test_objective(fn_type, fn_opts):
    '''Tests the objective specified by name.'''
 
    # Get fn opts
    opts = uf.get_fn_opts(fn_type, fn_opts)
    
    # Set some constants
    x_min = 0
    x_max = 1
    ndivs = 100
    ks = [1,2]
    
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
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    #ax.view_init(elev=25,azim=-35)
    surf = ax.plot_surface(x_mesh,y_mesh,z_mesh,cmap=plt.cm.viridis)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1,x_2)$')
    ax.set_title(f'Fn: {fn_type} | Options: {opts}')
    
    # Build print string
    print_string = '../figures/objectives/' + fn_type
    for value in opts.values():
        print_string += f'_{value}'
    print_string += '.png'
    
    return fig, print_string
    

def test_all_objectives():
    '''Tests all of the objectives.'''
    
    all_functions = uf.set_all_functions_default()
    all_functions = uf.set_functions(['sin2'])
    
    # Create pdf document
    pdf = PdfPages('../figures/objectives/plots_objectives_all.pdf')
    
    # Loop through all functions
    for fn, fn_opts in all_functions.items():
        
        # Test each combination of function options
        for opts in list(it.product(*fn_opts.values())):
            
            # Create the plot
            fig, print_string = test_objective(fn,opts)
            
            # Save the plot to file
            plt.savefig(print_string)
            
            # Save the plot to PDF
            pdf.savefig(fig)
            
    # Close the pdf
    pdf.close()
    
        
def get_objective(fn_type,fn_opts,degrees):
    '''Selects and returns a callable object of the correct function type.'''
    
    try:
        return getattr(Objective,fn_type.capitalize())(fn_opts,degrees)
    except:
        raise RuntimeError('Function type ' + fn_type + ' is not valid.')


if __name__ == '__main__':
    
    test_all_objectives()