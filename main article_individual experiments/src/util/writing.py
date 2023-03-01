# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:47:12 2022

@author: John Meluso
"""

import util.graphs as ug


def get_fns():
    
    # Weights
    unwt = '\\frac{1}{d}'
    unwt_half = '\\frac{1}{2d}'
    dgwt = '\\frac{1}{\sum_m \\left(k_m+1\\right)}'
    dgwt_half = '\\frac{1}{2\sum_m \\left(k_m+1\\right)}'
    
    return {
        'average_na_na_node': dict(
            name = 'Average function (unweighted)',
            equation = unwt + '\sum_m x_m'
            ),
        'average_na_na_degree': dict(
            name = 'Average function (degree-weighted)',
            equation = dgwt + '\sum_m \\left(k_m+1\\right)x_m'
            ),
        
        'sphere_na_na_node': dict(
            name = 'Average of squares function (unweighted)',
            equation = unwt + '\sum_m x_m^2'
            ),
        'sphere_na_na_degree': dict(
            name = 'Average of squares function (degree-weighted)',
            equation = dgwt + '\sum_m \\left(k_m+1\\right)x_m^2'
            ),
        
        'root_na_na_node': dict(
            name = 'Average of square roots function (unweighted)',
            equation = unwt + '\sum_m x_m^{1/2}'
            ),
        'root_na_na_degree': dict(
            name = 'Average of square roots function (degree-weighted)',
            equation = dgwt + '\sum_m \\left(k_m+1\\right)x_m^{1/2}'
            ),
        
        'min_na_na_na': dict(
            name = 'Minimum function',
            equation = 'min\\left({x_m}\\right)'
            ),
        'median_na_na_na': dict(
            name = 'Median function',
            equation = 'median\\left({x_m}\\right)'
            ),
        'max_na_na_na': dict(
            name = 'Maximum function',
            equation = 'max\\left({x_m}\\right)'
            ),
        
        'ackley_na_na_na': dict(
            name = 'Ackley function',
            equation = '1-\\frac{-c_1\exp\\left(-c_2\sqrt{\\frac{1}{k_i+1}\sum_m x_m^2}\\right)' \
                + '-\exp\\left(\\frac{1}{k_i+1}\sum_m \cos\\left(c_{3}x_m\\right)\\right)+exp\\left(1\\right)+c_1}' \
                + '{c_1\\left(1-exp\\left(-c_2\\right)\\right)+\\left(exp\\left(1\\right)-exp\\left(-1\\right)\\right)}',
            extra = 'For this implementation, we invert the Ackley function from its traditional concave form (for minimizing) to a convex form (for maximizing). Here, $c_1=20$, $c_2=0.2$, and $c_3=7\pi$.'
            ),
        
        'sin2_na_uniform_node': dict(
            name = '\sinsq function (unweighted, uniform frequency)',
            equation = unwt + '\sum_m sin^2\\left(\\frac{7\pi}{2}x_m\\right)'
            ),
        'sin2_na_degree_node': dict(
            name = '\sinsq function (unweighted, degree frequency)',
            equation = unwt + '\sum_m sin^2\\left(\\frac{\\left(1+2k_m\\right)\pi}{2}x_m\\right)'
            ),
        'sin2_na_uniform_degree': dict(
            name = '\sinsq function (degree-weighted, uniform frequency)',
            equation = dgwt + '\sum_m \\left(k_m+1\\right) sin^2\\left(\\frac{7\pi}{2}x_m\\right)'
            ),
        'sin2_na_degree_degree': dict(
            name = '\sinsq function (degree-weighted, degree frequency)',
            equation = dgwt + '\sum_m \\left(k_m+1\\right) sin^2\\left(\\frac{\\left(1+2k_m\\right)\pi}{2}x_m\\right)'
            ),
        
        'sin2sphere_na_uniform_node': dict(
            name = '\sinsq + square function (unweighted, uniform frequency)',
            equation = unwt_half + '\sum_m sin^2\\left(\\frac{7\pi}{2}x_m\\right) + x_m^2'
            ),
        'sin2sphere_na_degree_node': dict(
            name = '\sinsq + square function (unweighted, degree frequency)',
            equation = unwt_half + '\sum_m sin^2\\left(\\frac{\\left(1+2k_m\\right)\pi}{2}x_m\\right) + x_m^2'
            ),
        'sin2sphere_na_uniform_degree': dict(
            name = '\sinsq + square function (degree-weighted, uniform frequency)',
            equation = dgwt_half + '\sum_m \\left(k_m+1\\right)\\left(sin^2\\left(\\frac{7\pi}{2}x_m\\right) + x_m^2\\right)'
            ),
        'sin2sphere_na_degree_degree': dict(
            name = '\sinsq + square function (degree-weighted, degree frequency)',
            equation = dgwt_half + '\sum_m \\left(k_m+1\\right)\\left(sin^2\\left(\\frac{\\left(1+2k_m\\right)\pi}{2}x_m\\right) + x_m^2\\right)'
            ),
        
        'sin2root_na_uniform_node': dict(
            name = '\sinsq + square root function (unweighted, uniform frequency)',
            equation = unwt_half + '\sum_m sin^2\\left(\\frac{7\pi}{2}x_m\\right) + x_m^{1/2}'
            ),
        'sin2root_na_degree_node': dict(
            name = '\sinsq + square root function (unweighted, degree frequency)',
            equation = unwt_half + '\sum_m sin^2\\left(\\frac{\\left(1+2k_m\\right)\pi}{2}x_m\\right) + x_m^{1/2}'
            ),
        'sin2root_na_uniform_degree': dict(
            name = '\sinsq + square root function (degree-weighted, uniform frequency)',
            equation = dgwt_half + '\sum_m \\left(k_m+1\\right)\\left(sin^2\\left(\\frac{7\pi}{2}x_m\\right) + x_m^{1/2}\\right)'
            ),
        'sin2root_na_degree_degree': dict(
            name = '\sinsq + square root function (degree-weighted, degree frequency)',
            equation = dgwt_half + '\sum_m \\left(k_m+1\\right)\\left(sin^2\\left(\\frac{\\left(1+2k_m\\right)\pi}{2}x_m\\right) + x_m^{1/2}\\right)'
            ),
        
        'losqr_hiroot_uniform_na_node': dict(
            name = 'Low-degree square, high-degree root function (unweighted, uniform exponents)',
            equation = unwt + '\sum_m x_m^{2^{c_m}}',
            extra = 'Here, $c_m=1-\\frac{2k_1}{1 + \sum_m k_m}$.'
            ),
        'losqr_hiroot_degree_na_node': dict(
            name = 'Low-degree square, high-degree root function (unweighted, degree exponents)',
            equation = unwt + '\sum_m x_m^{2^{c_m}}',
            extra = 'Here, $c_m=1-\\frac{2k_m}{1 + \sum_m k_m}$.'
            ),
        'losqr_hiroot_uniform_na_degree': dict(
            name = 'Low-degree square, high-degree root function (degree-weighted, uniform exponents)',
            equation = dgwt + '\sum_m \\left(k_m+1\\right)x_m^{2^{c_m}}',
            extra = 'Here, $c_m=1-\\frac{2k_1}{1 + \sum_m k_m}$.'
            ),
        'losqr_hiroot_degree_na_degree': dict(
            name = 'Low-degree square, high-degree root function (degree-weighted, degee exponents)',
            equation = dgwt + '\sum_m \\left(k_m+1\\right)x_m^{2^{c_m}}',
            extra = 'Here, $c_m=1-\\frac{2k_m}{1 + \sum_m k_m}$.'
            ),
        
        'hisqr_loroot_uniform_na_node': dict(
            name = 'High-degree square, low-degree root function (unweighted, uniform exponents)',
            equation = unwt + '\sum_m x_m^{2^{-c_m}}',
            extra = 'Here, $c_m=1-\\frac{2k_1}{1 + \sum_m k_m}$.'
            ),
        'hisqr_loroot_degree_na_node': dict(
            name = 'High-degree square, low-degree root function (unweighted, degree exponents)',
            equation = unwt + '\sum_m x_m^{2^{-c_m}}',
            extra = 'Here, $c_m=1-\\frac{2k_m}{1 + \sum_m k_m}$.'
            ),
        'hisqr_loroot_uniform_na_degree': dict(
            name = 'High-degree square, low-degree root function (degree-weighted, uniform exponents)',
            equation = dgwt + '\sum_m \\left(k_m+1\\right)x_m^{2^{-c_m}}',
            extra = 'Here, $c_m=1-\\frac{2k_1}{1 + \sum_m k_m}$.'
            ),
        'hisqr_loroot_degree_na_degree': dict(
            name = 'High-degree square, low-degree root function (degree-weighted, degee exponents)',
            equation = dgwt + '\sum_m \\left(k_m+1\\right)x_m^{2^{-c_m}}',
            extra = 'Here, $c_m=1-\\frac{2k_m}{1 + \sum_m k_m}$.'
            ),
        
        'kth_power_na_na_node': dict(
            name = '\\texorpdfstring{$K+1$}{K+1} power function (unweighted)',
            equation = unwt + '\sum_m x_m^{k_m+1}'
            ),
        'kth_power_na_na_degree': dict(
            name = '\\texorpdfstring{$K+1$}{K+1} power function (degree-weighted)',
            equation = dgwt + '\sum_m \\left(k_m+1\\right)x_m^{k_m+1}'
            ),
        
        'kth_root_na_na_node': dict(
            name = '\\texorpdfstring{$K+1$}{K+1} Root function (unweighted)',
            equation = unwt + '\sum_m x_m^{\\frac{1}{k_m+1}}'
            ),
        'kth_root_na_na_degree': dict(
            name = '\\texorpdfstring{$K+1$}{K+1} Root function (degree-weighted)',
            equation = dgwt + '\sum_m \\left(k_m+1\\right)x_m^{\\frac{1}{k_m+1}}'
            ),
        }

def get_graphs():
    
    return {
        'complete_na_na_na': dict(
            name = 'Complete graph',
            label = 'Complete graph',
            graph_type = 'complete',
            graph_opts = ug.set_complete(),
            layout = 'circular',
            ),
        'empty_na_na_na':  dict(
            name = 'Empty graph (individual learning)',
            label = 'Empty graph',
            graph_type = 'empty',
            graph_opts = ug.set_empty(),
            layout = 'circular',
            ),
        'random_na_na_0.1':  dict(
            name = 'Random (\\texorpdfstring{$p=0.1$}{p=0.1})',
            label = 'Random ($p=0.1$)',
            graph_type = 'random',
            graph_opts = ug.set_random(p=0.1),
            layout = 'circular',
            ),
        'random_na_na_0.5':  dict(
            name = 'Random (\\texorpdfstring{$p=0.5$}{p=0.5})',
            label = 'Random ($p=0.5$)',
            graph_type = 'random',
            graph_opts = ug.set_random(p=0.5),
            layout = 'circular',
            ),
        'random_na_na_0.9':  dict(
            name = 'Random (\\texorpdfstring{$p=0.9$}{p=0.9})',
            label = 'Random ($p=0.9$)',
            graph_type = 'random',
            graph_opts = ug.set_random(p=0.9),
            layout = 'circular',
            ),
        'small_world_2_na_0.0':  dict(
            name = 'Ring',
            label = 'Ring',
            graph_type = 'small_world',
            graph_opts = ug.set_small_world(k=2,p=0.0),
            layout = 'circular',
            ),
        'small_world_2_na_0.1':  dict(
            name = 'Small world (\\texorpdfstring{$k=2$}{k=2}, \\texorpdfstring{$p=0.1$}{p=0.1})',
            label = 'Small world ($k=2$, $p=0.1$)',
            graph_type = 'small_world',
            graph_opts = ug.set_small_world(k=2,p=0.1),
            layout = 'circular',
            ),
        'small_world_2_na_0.5':  dict(
            name = 'Small world (\\texorpdfstring{$k=2$}{k=2}, \\texorpdfstring{$p=0.5$}{p=0.5})',
            label = 'Small world ($k=2$, $p=0.5$)',
            graph_type = 'small_world',
            graph_opts = ug.set_small_world(k=2,p=0.5),
            layout = 'circular',
            ),
        'small_world_2_na_0.9':  dict(
            name = 'Small world (\\texorpdfstring{$k=2$}{k=2}, \\texorpdfstring{$p=0.9$}{p=0.9})',
            label = 'Small world ($k=2$, $p=0.9$)',
            graph_type = 'small_world',
            graph_opts = ug.set_small_world(k=2,p=0.9),
            layout = 'circular',
            ),
        'power_na_2_0.1':  dict(
            name = 'Preferential attachment (\\texorpdfstring{$m=2$}{m=2}, \\texorpdfstring{$p=0.1$}{p=0.1})',
            label = 'Preferential attachment ($m=2$, $p=0.1$)',
            graph_type = 'power',
            graph_opts = ug.set_power(m=2,p=0.1),
            layout = 'kk',
            ),
        'power_na_2_0.5':  dict(
            name = 'Preferential attachment (\\texorpdfstring{$m=2$}{m=2}, \\texorpdfstring{$p=0.5$}{p=0.5})',
            label = 'Preferential attachment ($m=2$, $p=0.5$)',
            graph_type = 'power',
            graph_opts = ug.set_power(m=2,p=0.5),
            layout = 'kk',
            ),
        'power_na_2_0.9':  dict(
            name = 'Preferential attachment (\\texorpdfstring{$m=2$}{m=2}, \\texorpdfstring{$p=0.9$}{p=0.9})',
            label = 'Preferential attachment ($m=2$, $p=0.9$)',
            graph_type = 'power',
            graph_opts = ug.set_power(m=2,p=0.9),
            layout = 'kk',
            ),
        'ring_cliques_na_na_na':  dict(
            name = 'Ring of cliques',
            label = 'Ring of cliques',
            graph_type = 'ring_cliques',
            graph_opts = ug.set_ring_cliques(),
            layout = 'kk',
            ),
        'rook_na_na_na':  dict(
            name = 'Rook\'s graph',
            label = 'Rook\'s graph',
            graph_type = 'rook',
            graph_opts = ug.set_rook(),
            layout = 'grid',
            ),
        'star_na_na_na':  dict(
            name = 'Star',
            label = 'Star',
            graph_type = 'star',
            graph_opts = ug.set_star(),
            layout = 'kk',
            ),
        'tree_na_na_na':  dict(
            name = 'Tree',
            label = 'Tree',
            graph_type = 'tree',
            graph_opts = ug.set_tree(),
            layout = 'kk',
            ),
        'wheel_na_na_na':  dict(
            name = 'Wheel',
            label = 'Wheel',
            graph_type = 'wheel',
            graph_opts = ug.set_wheel(),
            layout = 'kk',
            ),
        'windmill_na_na_na':  dict(
            name = 'Windmill',
            label = 'Windmill',
            graph_type = 'windmill',
            graph_opts = ug.set_windmill(),
            layout = 'kk',
            ),
        }