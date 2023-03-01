# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:55:47 2022

@author: John Meluso
"""

# Import libraries
from dataclasses import dataclass, field

# Define simplified expressions
All = slice(None)

### Create Parent Variable Classes #########################################

@dataclass
class Levels:
    '''Creates levels of applicability for the specified variable.'''
    team: bool = False
    nbhd: bool = False
    agent: bool = False


@dataclass
class Variable:
    '''Creates a variable for model construction and/or analysis.'''
    dtype: type = object
    param_sim: bool = False
    param_model: bool = False
    descriptor: bool = False
    running: bool = False
    outcome: bool = False
    index: bool = False
    levels: Levels = Levels()
    default_slice: object = All

### Create Child Variable Classes ##########################################


@dataclass
class Model_type(Variable):
    dtype: type = str
    param_sim: bool = True
    param_model: bool = True
    index: bool = True
    levels: Levels = Levels(True, True, True)
    default_slice: object = All


@dataclass
class Team_size(Variable):
    dtype: type = int
    param_sim: bool = True
    param_model: bool = True
    index: bool = True
    levels: Levels = Levels(True, False, False)
    default_slice: object = All

@dataclass
class Team_graph_type(Variable):
    dtype: type = str
    param_sim: bool = True
    param_model: bool = True
    index: bool = True
    levels: Levels = Levels(True, False, False)
    default_slice: object = All

@dataclass
class Team_graph_opts(Variable):
    dtype: type = dict
    param_sim: bool = True
    param_model: bool = True
    levels: Levels = Levels(True, False, False)
    default_slice: object = All

@dataclass
class Team_fn_type(Variable):
    dtype: type = str
    param_sim: bool = True
    param_model: bool = True
    index: bool = True
    levels: Levels = Levels(True, False, False)
    default_slice: object = All

@dataclass
class Team_fn_opts(Variable):
    dtype: type = dict
    param_sim: bool = True
    param_model: bool = True
    levels: Levels = Levels(True, False, False)
    default_slice: object = All

@dataclass
class Nbhd_fn_type(Variable):
    dtype: type = str
    param_sim: bool = True
    param_model: bool = True
    index: bool = True
    levels: Levels = Levels(False, True, False)
    default_slice: object = All

@dataclass
class Nbhd_fn_opts(Variable):
    dtype: type = dict
    param_sim: bool = True
    param_model: bool = True
    levels: Levels = Levels(True, False, False)
    default_slice: object = All

@dataclass
class Agent_fn_type(Variable):
    dtype: type = str
    param_sim: bool = True
    param_model: bool = True
    index: bool = True
    levels: Levels = Levels(False, False, True)
    default_slice: object = All

@dataclass
class Agent_fn_opts(Variable):
    dtype: type = dict
    param_sim: bool = True
    param_model: bool = True
    levels: Levels = Levels(True, False, False)
    default_slice: object = All

@dataclass
class Agent_steplim(Variable):
    dtype: type = float
    param_sim: bool = True
    param_model: bool = True
    index: bool = True
    levels: Levels = Levels(True, True, True)
    default_slice: object = All

@dataclass
class Num_steps(Variable):
    dtype: type = int
    param_sim: bool = True
    levels: Levels = Levels(True, True, True)
    default_slice: object = All

@dataclass
class Run_step(Variable):
    dtype: type = int
    param_model: bool = True
    running: bool = True
    index: bool = True
    levels: Levels = Levels(True, True, True)
    default_slice: object = All

@dataclass
class Run_ind(Variable):
    dtype: type = int
    param_sim: bool = True
    param_model: bool = True
    index: bool = True
    default_slice: object = All

@dataclass
class Team_performance(Variable):
    dtype: type = float
    outcome: bool = True
    levels: Levels = Levels(True, False, False)

@dataclass
class Team_productivity(Variable):
    dtype: type = float
    outcome: bool = True
    levels: Levels = Levels(True, False, False)

@dataclass
class Team_graph_centrality_degree_mean(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)

@dataclass
class Team_graph_centrality_degree_stdev(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)

@dataclass
class Team_graph_centrality_eigenvector_mean(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)

@dataclass
class Team_graph_centrality_eigenvector_stdev(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)

@dataclass
class Team_graph_centrality_betweenness_mean(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)
    
@dataclass
class Team_graph_centrality_betweenness_stdev(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)
    
@dataclass
class Team_graph_nearest_neighbor_degree_mean(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)
    
@dataclass
class Team_graph_nearest_neighbor_degree_stdev(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)

@dataclass
class Team_graph_clustering(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)

@dataclass
class Team_graph_density(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)
    
@dataclass
class Team_graph_assortativity(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)

@dataclass
class Team_graph_pathlength(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)

@dataclass
class Team_graph_diameter(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)

@dataclass
class Team_fn_diff_integral(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)
    
@dataclass
class Team_fn_diff_peaks(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)

@dataclass
class Team_fn_alignment(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)

@dataclass
class Team_fn_interdep(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(True, False, False)

@dataclass
class Nbhd_fn_diff_integral(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(False, True, False)
    
@dataclass
class Nbhd_fn_diff_peaks(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(False, True, False)

@dataclass
class Agent_fn_diff_integral(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(False, False, True)
    
@dataclass
class Agent_fn_diff_peaks(Variable):
    dtype: type = float
    descriptor: bool = True
    levels: Levels = Levels(False, False, True)

@dataclass
class Team_graph_k(Variable):
    dtype: type = object
    index: bool = True
    levels: Levels = Levels(True, False, False)
    default_slice: object = field(default_factory=lambda: ['na',str(2)])

@dataclass
class Team_graph_m(Variable):
    dtype: type = object
    index: bool = True
    levels: Levels = Levels(True, False, False)
    default_slice: object = field(default_factory=lambda: ['na',str(2)])

@dataclass
class Team_graph_p(Variable):
    dtype: type = object
    index: bool = True
    levels: Levels = Levels(True, False, False)
    default_slice: object = field(default_factory=lambda: ['na',str(0.5)])

@dataclass
class Team_fn_weight(Variable):
    dtype: type = str
    index: bool = True
    levels: Levels = Levels(True, False, False)
    default_slice: object = field(default_factory=lambda: ['na','node'])

@dataclass
class Team_fn_frequency(Variable):
    dtype: type = str
    index: bool = True
    levels: Levels = Levels(True, False, False)
    default_slice: object = field(default_factory=lambda: ['na','uniform'])

@dataclass
class Team_fn_exponent(Variable):
    dtype: type = str
    index: bool = True
    levels: Levels = Levels(True, False, False)
    default_slice: object = field(default_factory=lambda: ['na','degree'])

@dataclass
class Nbhd_fn_weight(Variable):
    dtype: type = str
    index: bool = True
    levels: Levels = Levels(False, True, False)
    default_slice: object = field(default_factory=lambda: ['na','node'])

@dataclass
class Nbhd_fn_frequency(Variable):
    dtype: type = str
    index: bool = True
    levels: Levels = Levels(False, True, False)
    default_slice: object = field(default_factory=lambda: ['na','uniform'])

@dataclass
class Nbhd_fn_exponent(Variable):
    dtype: type = str
    index: bool = True
    levels: Levels = Levels(False, True, False)
    default_slice: object = field(default_factory=lambda: ['na','degree'])

@dataclass
class Agent_fn_weight(Variable):
    dtype: type = str
    index: bool = True
    levels: Levels = Levels(False, False, True)
    default_slice: object = field(default_factory=lambda: ['na','node'])

@dataclass
class Agent_fn_frequency(Variable):
    dtype: type = str
    index: bool = True
    levels: Levels = Levels(False, False, True)
    default_slice: object = field(default_factory=lambda: ['na','uniform'])

@dataclass
class Agent_fn_exponent(Variable):
    dtype: type = str
    index: bool = True
    levels: Levels = Levels(False, False, True)
    default_slice: object = field(default_factory=lambda: ['na','degree'])
