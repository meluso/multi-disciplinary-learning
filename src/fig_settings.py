# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:11:05 2021

@author: John Meluso
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

def set_fonts(extra_params={}):
    params = {
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'cm',
        'legend.fontsize': 8,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.titlesize': 8
        }
    for key, value in extra_params.items():
        params[key] = value
    pylab.rcParams.update(params)
    
def fig_size(frac_width,frac_height,n_cols=1,n_rows=1):
    
    # Set default sizes
    page_width = 8.5
    page_height = 11
    side_margins = 1
    tb_margins = 1
    middle_margin = 0.25
    mid_marg_width = middle_margin*(n_cols-1)
    mid_marg_height = middle_margin*(n_rows-1)
    
    # Width logic
    if frac_width == 1:
        width = page_width - side_margins
    else:
        width = (page_width - side_margins - mid_marg_width)*frac_width
        
    # Height logic
    if frac_height == 1:
        height = page_height - tb_margins
    else:
        height = (page_height - tb_margins - mid_marg_height)*frac_height
        
    return (width,height)

def get_formats():
    return ['eps','jpg','pdf','png','tif']

def set_border(ax, top=False, bottom=False, left=False, right=False):
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)

def save_pub_fig(name, dpi=1200, **kwargs):
    for ff in get_formats():
        fname = f'../figures/publication/{ff}/{name}.{ff}'
        plt.savefig(fname, format=ff, dpi=dpi, **kwargs)
