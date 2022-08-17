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
        self.model_type = ['1x','1f','2x','2f','3xx','3fx','3xg','3fg']

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
            # 'complete': ug.set_complete(),
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
        fns = ['sin2sphere','median','ackley']
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
            'ring_cliques': ug.set_ring_cliques(),
            'rook': ug.set_rook(),
            'small_world': ug.set_small_world(p=[0,0.1,0.5,0.9]),
            'star': ug.set_star(),
            'tree': ug.set_tree(),
            'wheel': ug.set_wheel(),
            'windmill': ug.set_windmill()
            }
        self.nbhd_fn2opt = uf.set_all_functions_default()
        
        # Neighborhood parameters
        self.nbhd_fn2opt = uf.set_all_functions_default()
        
        # Agent parameters
        self.agent_fn2opt = uf.set_agent_passthrough()
        self.agent_steplim = [1.0]
        
        # Running parameters
        self.num_steps = [5]
        self.num_runs = [2]
    

class Params2x(Parameters):
    
    def __init__(self):
        '''Default parameters for Model 2x.'''
        super().__init__()
        
        # Model parameters
        self.model_type = ['2x']
        
        # Neighborhood parameters
        self.nbhd_fn2opt = uf.set_all_functions_off()
        
        # Agent parameters
        self.agent_fn2opt = uf.set_agent_passthrough()
    
    
class Params2f(Parameters):
    
    def __init__(self):
        '''Default parameters for Model 2f.'''
        super().__init__()
        
        # Model parameters
        self.model_type = ['2f']
        
        # Neighborhood parameters
        self.nbhd_fn2opt = uf.set_all_functions_off()
    
    
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
        
        
class Params004(Parameters):
    
    def __init__(self):
        '''Parameter Set 004'''
        super().__init__()
        
        # Model parameters
        self.model_type = ['2x']
        
        # Neighborhood parameters
        self.nbhd_fn2opt = uf.set_all_functions_off()
        

class Params005(Parameters):
    
    def __init__(self):
        '''Parameter Set 005'''
        super().__init__()
        
        # Model parameters
        self.model_type = ['2f']
        
        # Neighborhood parameters
        self.nbhd_fn2opt = uf.set_all_functions_off()
        
        
class Params006(Parameters):
    
    def __init__(self):
        '''Parameter Set 006'''
        super().__init__()        
        
        # Model parameters
        self.model_type = ['3xx']
        
        # Agent parameters
        self.agent_fn2opt = uf.set_agent_passthrough()
        

class Params007(Parameters):
    
    def __init__(self):
        '''Parameter Set 007'''
        super().__init__()        
        
        # Model parameters
        self.model_type = ['3fg']
        
        # Agent parameters
        self.agent_fn2opt = uf.set_agent_passthrough()


class Params011(Parameters):
    
    def __init__(self):
        super().__init__()
        
        # Model parameters
        self.model_type = ['2x','2f']
        
        # Team parameters
        self.team_size = [5, 10, 25]
        
        # Neighborhood parameters
        self.nbhd_fn2opt = uf.set_all_functions_off()


class Params012(Parameters):
    
    def __init__(self):
        super().__init__()
        
        # Model parameters
        self.model_type = ['2x','2f']
        
        # Team parameters
        self.team_size = [50]
        
        # Neighborhood parameters
        self.nbhd_fn2opt = uf.set_all_functions_off()

        
class Params013(Parameters):
    
    def __init__(self):
        super().__init__()
        
        # Model parameters
        self.model_type = ['2x']
        
        # Team parameters
        self.team_size = [5, 10, 25]
        
        # Neighborhood parameters
        self.nbhd_fn2opt = uf.set_all_functions_off()
        
        # Agent parameters
        self.agent_fn2opt = uf.set_agent_passthrough()
        
        # Running parameters
        self.num_runs = [10000]
        
        
class Params021(Parameters):
    
    def __init__(self):
        super().__init__()        
        
        # Model parameters
        self.model_type = ['3xx','3fg']
        
        # Team parameters
        self.team_size = [5, 10, 25]
        
        # Agent parameters
        self.agent_fn2opt = uf.set_agent_passthrough()
        
        
class Params022(Parameters):
    
    def __init__(self):
        super().__init__()        
        
        # Model parameters
        self.model_type = ['3xx','3fg']
        
        # Team parameters
        self.team_size = [50]
        
        # Agent parameters
        self.agent_fn2opt = uf.set_agent_passthrough()
        
        
class Params023(Parameters):
    
    def __init__(self):
        super().__init__()        
        
        # Model parameters
        self.model_type = ['3xx']
        
        # Team parameters
        self.team_size = [5, 10, 25]
        
        # Agent parameters
        self.agent_fn2opt = uf.set_agent_passthrough()
        
        
class Params024(Parameters):
    
    def __init__(self):
        super().__init__()        
        
        # Model parameters
        self.model_type = ['3fx']
        
        # Team parameters
        self.team_size = [5, 10, 25]
        
        # Agent parameters
        self.agent_fn2opt = uf.set_agent_passthrough()
        

class Params025(Parameters):
    
    def __init__(self):
        super().__init__()        
        
        # Model parameters
        self.model_type = ['3xg']
        
        # Team parameters
        self.team_size = [5, 10, 25]
        
        # Agent parameters
        self.agent_fn2opt = uf.set_agent_passthrough()
        
        
class Params026(Parameters):
    
    def __init__(self):
        super().__init__()        
        
        # Model parameters
        self.model_type = ['3fg']
        
        # Team parameters
        self.team_size = [5, 10, 25]
        
        # Agent parameters
        self.agent_fn2opt = uf.set_agent_passthrough()
        

class Params031(Params2x):
    
    def __init__(self):
        super().__init__()
        
        # Agent parameters
        self.agent_steplim = [1.0]
        

class Params032(Params3xx):
    
    def __init__(self):
        super().__init__()
        
        # Agent parameters
        self.agent_steplim = [1.0]


class Params033(Params3xg):
    
    def __init__(self):
        super().__init__()
        
        # Agent parameters
        self.agent_steplim = [1.0]

        
class Params034(Params3xx):
    
    def __init__(self):
        super().__init__()
        
        # Team parameters
        self.team_graph2opt['small_world'] \
            = ug.set_small_world(p=[0,0.1,0.5,0.9])
        
        # Agent parameters
        self.agent_steplim = [1.0]


class Params035(Params3xg):
    
    def __init__(self):
        super().__init__()
        
        # Team parameters
        self.team_graph2opt['small_world'] \
            = ug.set_small_world(p=[0,0.1,0.5,0.9])
        
        # Agent parameters
        self.agent_steplim = [1.0]
        
        
class Params041(Params3xx):
    
    def __init__(self):
        super().__init__()
        
        # Agent parameters
        self.agent_steplim = [0.01]
        
        # Running parameters
        self.num_runs = [250]


class Params042(Params3xx):
    
    def __init__(self):
        super().__init__()
        
        # Agent parameters
        self.agent_steplim = [0.10]
        
        # Running parameters
        self.num_runs = [250]
        
        
class Params043(Params3xx):
    
    def __init__(self):
        super().__init__()
        
        # Agent parameters
        self.agent_steplim = [1.00]
        
        # Running parameters
        self.num_runs = [250]


class Params044(Params3xx):
    
    def __init__(self):
        super().__init__()
        
        # Agent parameters
        self.agent_steplim = [0.001]
        
        # Running parameters
        self.num_runs = [250]
        

class Params045(Params3xx):
    
    def __init__(self):
        super().__init__()
        
        # Agent parameters
        self.agent_steplim = [0.001]
        
        # Running parameters
        self.num_runs = [250]
        
        
class Params046(Params3xx):
    
    def __init__(self):
        super().__init__()
        
        # Agent parameters
        self.agent_steplim = [0.010]
        
        # Running parameters
        self.num_runs = [250]
        
        
class Params047(Params3xx):
    
    def __init__(self):
        super().__init__()
        
        # Agent parameters
        self.agent_steplim = [0.100]
        
        # Running parameters
        self.num_runs = [250]
        
        
class Params048(Params3xx):
    
    def __init__(self):
        super().__init__()
        
        # Agent parameters
        self.agent_steplim = [1.000]
        
        # Running parameters
        self.num_runs = [250]