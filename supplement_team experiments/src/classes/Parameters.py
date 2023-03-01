# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:49:11 2021

@author: John Meluso
"""

# Import libraries
from itertools import product as pd
import numpy as np

# Import model functions
import util.functions as uf
import util.graphs as ug
import util.variables as uv


class Parameters(object):
    
    def __init__(self):
        '''Initializes an instances of a set of possible values to run during a
        simulation of a team.'''
        
        # Model parameters
        self.model_type = ['3xx','3fx','3xg','3fg']

        # Team parameters
        self.team_size = [4, 9, 16, 25]
        self.team_graph2opt = {
            'complete': ug.set_complete(),
            'empty': ug.set_empty(),
            'power': ug.set_power(p=(0.1,0.9,3)),
            'random': ug.set_random(p=(0.1,0.9,3)),
            'ring_cliques': ug.set_ring_cliques(),
            'rook': ug.set_rook(),
            'small_world': ug.set_small_world(p=[0,0.1,0.5,0.9]),
            'star': ug.set_star(),
            'tree': ug.set_tree(),
            'wheel': ug.set_wheel(),
            'windmill': ug.set_windmill()
            }
        self.team_fn2opt = uf.set_all_functions_default()
        
        # Neighborhood parameters
        self.nbhd_fn2opt = uf.set_all_functions_default()
        
        # Agent parameters
        self.agent_fn2opt = uf.set_all_functions_default()
        self.agent_steplim = np.round(np.linspace(0.1,1,4),decimals=3)
        
        # Running parameters
        self.num_steps = [25]
        self.num_runs = [100]
        
    def iter_all(self, rn, tgt, tft, nft, aft):
        '''Build iterator of all iterators.'''
        return pd(range(rn), self.iter_team_graph(tgt), self.iter_team_fn(tft),
                  self.iter_nbhd_fn(nft), self.iter_agent_fn(aft))
        
    def iter_team_graph(self, tgt):
        '''Build iterator for team graph options.'''
        #return list(pd(*self.team_graph2opt[tgt].values()))
        return ug.product_dict(**self.team_graph2opt[tgt])

    def iter_team_fn(self, tft):
        '''Build iterator for team function options.'''
        return list(pd(*self.team_fn2opt[tft].values()))

    def iter_nbhd_fn(self, nft):
        '''Build iterator for neighborhood function options.'''
        return list(pd(*self.nbhd_fn2opt[nft].values()))

    def iter_agent_fn(self, aft):
        '''Build iterator for agent function options.'''
        return list(pd(*self.agent_fn2opt[aft].values()))
    
    def build_params(self, all_cases=False, get_cases=[]):
        '''Builds the parameters from the set of possible options.'''
        
        # Create empty list of all parameter combinations
        params = []
        
        count = -1
        
        for mt, ts, tgt, tft, nft, aft, asl, ns, rn \
            in list(pd(*self.__dict__.values())):
            
            for run,tgo,tfo,nfo,afo in self.iter_all(rn,tgt,tft,nft,aft):
                            
                count += 1
                
                # Build new parameter set
                new_param = uv.get_param_sim(
                    model_type=mt,
                    team_size=ts,
                    team_graph_type=tgt,
                    team_graph_opts=tgo,
                    team_fn_type=tft,
                    team_fn_opts=uf.get_fn_opts(tft,tfo),
                    nbhd_fn_type=nft,
                    nbhd_fn_opts=uf.get_fn_opts(nft,nfo),
                    agent_fn_type=aft,
                    agent_fn_opts=uf.get_fn_opts(aft,afo),
                    agent_steplim=asl,
                    num_steps=ns,
                    run_ind=run
                    )
                
                # Append parameter set to all params
                if all_cases or (count in get_cases):
                    params.append(new_param)
        
        return count, params


class ParamsTestOnce(Parameters):
    
    def __init__(self):
        '''Initialize a test instance of parameters.'''
        super().__init__()
        
        # Model parameters
        self.model_type = ['3xx']
        
        # Team parameters
        self.team_size = [25]
        self.team_graph2opt = {
            'complete': ug.set_complete(),
            # 'empty': ug.set_empty(),
            'power': ug.set_power(p=(0.1,0.9,3)),
            'random': ug.set_random(p=(0.1,0.9,3)),
            # 'ring_cliques': ug.set_ring_cliques(),
            # 'rook': ug.set_rook(),
            'small_world': ug.set_small_world(p=[0,0.1,0.5,0.9]),
            # 'star': ug.set_star(),
            # 'tree': ug.set_tree(),
            # 'wheel': ug.set_wheel(),
            # 'windmill': ug.set_windmill()
            }
        fns = ['ackley','average','sin2sphere']
        self.team_fn2opt = uf.set_functions(fns)
        
        # Neighborhood parameters
        self.nbhd_fn2opt = uf.set_functions(fns)
        
        # Agent parameters
        self.agent_fn2opt = uf.set_agent_passthrough()
        self.agent_steplim = [1.0]
        
        # Running parameters
        self.num_steps = [25]
        self.num_runs = [3]

        
class ParamsTestModels(Parameters):
    
    def __init__(self):
        '''Initialize a larger set of tests parameters.'''
        super().__init__()
        
        # Model parameters
        self.model_type = ['3xx']
        
        # Team parameters
        self.team_size = [9]
        self.team_graph2opt = {
            'complete': ug.set_complete(),
            'empty': ug.set_empty(),
            'power': ug.set_power(p=0.5),
            'random': ug.set_random(p=0.5),
            # 'ring_cliques': ug.set_ring_cliques(),
            # 'rook': ug.set_rook(),
            'small_world': ug.set_small_world(p=[0,0.1,0.5,0.9]),
            # 'star': ug.set_star(),
            # 'tree': ug.set_tree(),
            # 'wheel': ug.set_wheel(),
            # 'windmill': ug.set_windmill()
            }
        fns = ['average','ackley','losqr_hiroot','kth_root','kth_power']
        self.team_fn2opt = uf.set_functions(fns)
        
        # Neighborhood parameters
        self.nbhd_fn2opt = uf.set_functions(fns)
        
        # Agent parameters
        self.agent_fn2opt = uf.set_agent_passthrough()
        self.agent_steplim = [0.001, 0.01, 0.1, 1.0]
        
        # Running parameters
        self.num_steps = [15]
        self.num_runs = [5]
    
    
class Params3xx(Parameters):
    
    def __init__(self):
        '''Default parameters for Model 3xx.'''
        super().__init__()
        
        # Model parameters
        self.model_type = ['3xx']
        
        # Agent parameters
        self.agent_fn2opt = uf.set_agent_passthrough()
    
    
class Params3fx(Parameters):
    
    def __init__(self):
        '''Default parameters for Model 3fx.'''
        super().__init__()
        
        
        # Model parameters
        self.model_type = ['3fx']
    
    
class Params3xg(Parameters):
    
    def __init__(self):
        '''Default parameters for Model 3xg.'''
        super().__init__()
        
        # Model parameters
        self.model_type = ['3xg']
        
        # Agent parameters
        self.agent_fn2opt = uf.set_agent_passthrough()
    
    
class Params3fg(Parameters):
    
    def __init__(self):
        '''Default parameters for Model 3fg.'''
        super().__init__()
        
        # Model parameters
        self.model_type = ['3fg']
        

class Params001(Params3xx):
    
    def __init__(self):
        super().__init__()
        
        # Agent parameters
        self.agent_steplim = [
            0.001,
            0.01,
            0.025, 0.050, 0.075, 0.1,
            0.25, 0.5, 0.75, 1.0
            ]
        
        # Running parameters
        self.num_runs = [50]
        self.num_steps = [50]