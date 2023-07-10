# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:23:34 2022

@author: John Meluso
"""

# Import model files
import util.analysis as ua

def arrow(ax, xyfrom, xyto, text=''):
    an = ax.annotate(text=text, xy=xyto, xytext=xyfrom, annotation_clip=False,
        arrowprops=dict(arrowstyle='->',fc='#AAAAAA',ec='#AAAAAA'),
        xycoords='axes fraction')
    return an

def level_indices(dataframe):
    '''Get dictionary of indices of all variables in dataframe.'''

    # Create dictionary for variable indices
    variables = {}

    # Iterate through each level and save index
    for ii, level in enumerate(dataframe.index.levels):
        variables[level.name] = {}
        variables[level.name]['values'] = level.array
        variables[level.name]['index'] = ii

    return variables

def drop_vars(model):
    '''Sets variables to drop for confidence interval plotting.'''

    if model == '2x':
        keep = [
            'model_type',
            'team_size',
            'team_graph_type',
            'team_fn_type',
            'run_step'
            ]
    elif model == '2f':
        keep = [
            'model_type',
            'team_size',
            'team_graph_type',
            'team_fn_type',
            'agent_fn_type',
            'run_step'
            ]
    elif model == '3xx':
        keep = [
            'model_type',
            'team_size',
            'team_graph_type',
            'team_fn_type',
            'nbhd_fn_type',
            'run_step'
            ]
    elif model == '3fx':
        keep = [
            'model_type',
            'team_size',
            'team_graph_type',
            'team_fn_type',
            'nbhd_fn_type',
            'run_step'
            ]
    elif model == '3xg':
        keep = [
            'model_type',
            'team_size',
            'team_graph_type',
            'team_fn_type',
            'nbhd_fn_type',
            'run_step'
            ]
    elif model == '3fg':
        keep = [
            'model_type',
            'team_size',
            'team_graph_type',
            'team_fn_type',
            'nbhd_fn_type',
            'run_step'
            ]
    else:
        raise RuntimeError(f'Model {model} is not valid.')

    # Pare down to drop vars
    var_list = [var for var in ua.group_vars(model) if var not in keep]

    return var_list

def get_graph_labels():

    return {
        'complete_na_na_na': 'Complete',
        'empty_na_na_na': 'Empty (Indiv. Learning)',
        'power_na_2_0.1': 'Pref. Attach. ($m=2$, $p=0.1$)',
        'power_na_2_0.5': 'Pref. Attach. ($m=2$, $p=0.5$)',
        'power_na_2_0.9': 'Pref. Attach. ($m=2$, $p=0.9$)',
        'random_na_na_0.1': 'Random ($p=0.1$)',
        'random_na_na_0.5': 'Random ($p=0.5$)',
        'random_na_na_0.9': 'Random ($p=0.9$)',
        'ring_cliques_na_na_na': 'Ring of Cliques',
        'rook_na_na_na': 'Rook\'s Graph',
        'small_world_2_na_0.0': 'Ring',
        'small_world_2_na_0.1': 'Small World ($k=2$, $p=0.1$)',
        'small_world_2_na_0.5': 'Small World ($k=2$, $p=0.5$)',
        'small_world_2_na_0.9': 'Small World ($k=2$, $p=0.9$)',
        'star_na_na_na': 'Star',
        'tree_na_na_na': 'Tree',
        'wheel_na_na_na': 'Wheel',
        'windmill_na_na_na': 'Windmill'
        }

def get_fn_labels():

    return {
        'ackley_na_na_na': 'Ackley',
        'average_na_na_degree': 'Average (degree-wt.)',
        'average_na_na_node': 'Average (unwt.)',
        'hisqr_loroot_degree_na_degree': 'High Square-Low Root (degree-wt., degee exp.)',
        'hisqr_loroot_degree_na_node': 'High Square-Low Root (unwt., degree exp.)',
        'hisqr_loroot_uniform_na_degree': 'High Square-Low Root (degree-wt., uniform exp.)',
        'hisqr_loroot_uniform_na_node': 'High Square-Low Root (unwt., uniform exp.)',
        'kth_power_na_na_degree': '$K+1$ Power (degree-wt.)',
        'kth_power_na_na_node': '$K+1$ Power (unwt.)',
        'kth_root_na_na_degree': '$K+1$ Root (degree-wt.)',
        'kth_root_na_na_node': '$K+1$ Root (unwt.)',
        'losqr_hiroot_degree_na_degree': 'Low Square-High Root (degree-wt., degee exp.)',
        'losqr_hiroot_degree_na_node': 'Low Square-High Root (unwt., degree exp.)',
        'losqr_hiroot_uniform_na_degree': 'Low Square-High Root (degree-wt., uniform exp.)',
        'losqr_hiroot_uniform_na_node': 'Low Square-High Root (unwt., uniform exp.)',
        'max_na_na_na': 'Maximum',
        'median_na_na_na': 'Median',
        'min_na_na_na': 'Minimum',
        'root_na_na_degree': 'Ave. of Square Roots (degree-wt.)',
        'root_na_na_node': 'Ave. of Square Roots (unwt.)',
        'sin2_na_degree_degree': 'Sin$^2$ (degree-wt., degree freq.)',
        'sin2_na_degree_node': 'Sin$^2$ (unwt., degree freq.)',
        'sin2_na_uniform_degree': 'Sin$^2$ (degree-wt., uniform freq.)',
        'sin2_na_uniform_node': 'Sin$^2$ (unwt., uniform freq.)',
        'sin2root_na_degree_degree': 'Sin$^2$ + Sq. Rt. (degree-wt., degree freq.)',
        'sin2root_na_degree_node': 'Sin$^2$ + Sq. Rt. (unwt., degree freq.)',
        'sin2root_na_uniform_degree': 'Sin$^2$ + Sq. Rt. (degree-wt., uniform freq.)',
        'sin2root_na_uniform_node': 'Sin$^2$ + Sq. Rt. (unwt., uniform freq.)',
        'sin2sphere_na_degree_degree': 'Sin$^2$ + Square (degree-wt., degree freq.)',
        'sin2sphere_na_degree_node': 'Sin$^2$ + Square (unwt., degree freq.)',
        'sin2sphere_na_uniform_degree': 'Sin$^2$ + Square (degree-wt., uniform freq.)',
        'sin2sphere_na_uniform_node': 'Sin$^2$ + Square (unwt., uniform freq.)',
        'sphere_na_na_degree': 'Ave. of Squares (degree-wt.)',
        'sphere_na_na_node': 'Ave. of Squares (unwt.)'
        }

def get_metric_labels():

    return {
        'team_graph_centrality_degree_mean': 'Degree Cent. (Mean)',
        'team_graph_centrality_degree_stdev': 'Degree Cent. (St. Dev.)',
        'team_graph_centrality_eigenvector_mean': 'Eigenvector Cent. (Mean)',
        'team_graph_centrality_eigenvector_stdev': 'Eigenvector Cent. (St. Dev.)',
        'team_graph_centrality_betweenness_mean': 'Betweenness Cent. (Mean)',
        'team_graph_centrality_betweenness_stdev': 'Betweenness Cent. (St. Dev.)',
        'team_graph_nearest_neighbor_degree_mean': 'Nearest Neighbor Degree (Mean)',
        'team_graph_nearest_neighbor_degree_stdev': 'Nearest Neighbor Degree (St. Dev.)',
        'team_graph_clustering': 'Clustering Coeff.',
        'team_graph_assortativity': 'Degree Assortativity',
        'team_graph_pathlength': 'Shortest Path Length (Mean)',
        'team_graph_diameter': 'Diameter',
        'team_fn_diff_integral': 'Exploration difficulty (1 - Task integral)',
        'log10(team_fn_diff_peaks)': 'log$_{10}$(Task number of peaks)',
        'team_fn_diff_peaks': 'Exploitation difficulty (Number of peaks)',
        'team_fn_alignment': 'Neighborhood alignment',
        'team_fn_interdep': 'Neighborhood interdependence'
        }

def get_metrics():

    return {
        'team_graph_centrality_degree_mean': {
            'plot': False,
            'meas_type': 'conn',
            'label': 'Well-connected\n(deg. cent. mean)',
            'annot_loc': (-5, -5),
            'use_conn': False,
            'ha': 'center',
            'va': 'top'
            },
        'team_graph_centrality_degree_stdev': {
            'plot': False,
            'meas_type': 'conn',
            'label': 'Varied connectedness\n(deg. cent. st. dev.)',
            'annot_loc': (5, -5),
            'use_conn': False,
            'ha': 'left',
            'va': 'center'
            },
        'team_graph_centrality_eigenvector_mean': {
            'plot': True,
            'meas_type': 'conn_to_conn',
            'label': 'Decentralization\n(eig. cent. mean)',
            'annot_loc': (-2.5, 5),
            'use_conn': False,
            'ha': 'center',
            'va': 'bottom'
            },
        'team_graph_centrality_eigenvector_stdev': {
            'plot': False,
            'meas_type': 'conn_to_conn',
            'label': 'Varied\ncentralization\n(eig. cent. st. dev.)',
            'annot_loc': (-5, 5),
            'use_conn': False,
            'ha': 'center',
            'va': 'bottom'
            },
        'team_graph_centrality_betweenness_mean': {
            'plot': True,
            'meas_type': 'efficiency',
            'label': 'Many intermediaries\n(bet. cent. mean)',
            'annot_loc': (-20, -20),
            'use_conn': True,
            'ha': 'center',
            'va': 'top'
            },
        'team_graph_centrality_betweenness_stdev': {
            'plot': True,
            'meas_type': 'efficiency',
            'label': 'Varied\nintermediarity\n(bet. cent. st. dev.)',
            'annot_loc': (-20, 30),
            'use_conn': True,
            'ha': 'center',
            'va': 'bottom'
            },
        'team_graph_nearest_neighbor_degree_mean': {
            'plot': False,
            'meas_type': 'conn_to_conn',
            'label': 'Well-connected\nneighbors\n(NND mean)',
            'annot_loc': (-7, -8),
            'use_conn': False,
            'ha': 'right',
            'va': 'center'
            },
        'team_graph_nearest_neighbor_degree_stdev': {
            'plot': True,
            'meas_type': 'conn_to_conn',
            'label': 'Varied neighbor\nconnectedness\n(NND st. dev.)',
            'annot_loc': (0, -5),
            'use_conn': False,
            'ha': 'center',
            'va': 'top'
            },
        'team_graph_clustering': {
            'plot': False,
            'meas_type': 'grouping',
            'label': "To neighbors'\nneighbors\n(clust. coeff.)",
            'annot_loc': (0, -5),
            'use_conn': False,
            'ha': 'center',
            'va': 'top'
            },
        'team_graph_assortativity': {
            'plot': False,
            'meas_type': 'grouping',
            'label': 'To the similarly\nconnected\n(deg. assort.)',
            'annot_loc': (-5, 30),
            'use_conn': True,
            'ha': 'center',
            'va': 'bottom'
            },
        'team_graph_pathlength': {
            'plot': False,
            'meas_type': 'efficiency',
            'label': 'Longer paths\n(shortest path\nlength mean)',
            'annot_loc': (-5, 0),
            'use_conn': False,
            'ha': 'right',
            'va': 'center'
            },
        'team_graph_diameter': {
            'plot': False,
            'meas_type': 'efficiency',
            'label': 'Longest path\n(diameter)',
            'annot_loc': (0, -5),
            'use_conn': False,
            'ha': 'center',
            'va': 'top'
            },
        'team_fn_diff_integral': {
            'plot': True,
            'meas_type': 'task',
            'label': 'Exploration difficulty\n(1 - task integral)',
            'annot_loc': (-5, 0),
            'use_conn': False,
            'ha': 'right',
            'va': 'center'
            },
        'team_fn_diff_peaks': {
            'plot': True,
            'meas_type': 'task',
            'label': 'Exploitation difficulty\n(num. of peaks)',
            'annot_loc': (0, 5),
            'use_conn': False,
            'ha': 'center',
            'va': 'bottom'
            },
        'team_fn_alignment': {
            'plot': True,
            'meas_type': 'task',
            'label': 'Neighborhood\nalignment',
            'annot_loc': (-5, 0),
            'use_conn': False,
            'ha': 'right',
            'va': 'center'
            },
        'team_fn_interdep': {
            'plot': True,
            'meas_type': 'task',
            'label': 'Neighborhood\ninterdependence',
            'annot_loc': (5, 5),
            'use_conn': False,
            'ha': 'left',
            'va': 'center'
            }
        }

def get_measure_formatting():
    
    return {
        'conn': {
            'color': '#AA0000',
            'marker': 'o',
            'label': "(d) Individuals' connectedness",
            'pres-label': "Individuals' connectedness"
            },
        'conn_to_conn': {
            'color': '#39AA00',
            'marker': 'D',
            'label': "(a) Neighbors' connectedness",
            'pres-label': "Neighbors' connectedness",
            },
        'efficiency': {
            'color': '#3900AA',
            'marker': 'H',
            'label': "(b) Network efficiency",
            'pres-label': "Network efficiency",
            },
        'grouping': {
            'color': '#AA8E00',
            'marker': 's',
            'label': "(c) Shared connections",
            'pres-label': "Shared connections",
            },
        'task': {
            'color': '#AA008E',
            'marker': 'X',
            'label': "Task measures",
            'pres-label': "Task measures",
            },
        }
