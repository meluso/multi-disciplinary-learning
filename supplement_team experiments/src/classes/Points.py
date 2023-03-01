# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 19:44:32 2021

@author: John Meluso
"""

# Import libraries
import pandas as pd

# Import model methods
import util.data as udata
import util.variables as uv


class Points(object):
    
    def __init__(self):
        '''Initialize points storage structure as a dictionary.'''
        
        # Create parameter attributes
        for key, value in uv.get_param_model().items():
            setattr(self, key, value)
            
        # Create descriptor attributes
        for key, value in uv.get_descriptors().items():
            setattr(self, key, value)
            
        # Create outcome attributes
        for key, value in uv.get_outcomes().items():
            setattr(self, key, value)
        
        # Create running attributes
        for key, value in uv.get_runnings().items():
            setattr(self, key, value)
            
        
    def update(self, params, team, model_type, run_ind, step):
        
        # Update model values
        self.model_type.append(model_type)
        
        # Append each parameter
        for key, value in params.items():
            self.__dict__[key].append(value)
        
        # Update running values
        self.run_ind.append(run_ind)
        self.run_step.append(step)
        
        # Update output values
        self.team_performance.append(team.get_team_fx())
        self.team_productivity.append(team.get_team_dfdt())
        
        # Append descriptor values
        for key in uv.get_descriptors():
            self.__dict__[key].append(getattr(team, f'get_{key}')())
        
        
    def get_dict(self):
        
        # Construct NON-OPTION points from simple fields
        points = self.__dict__.copy()
        
        # Create dictionaries for OPTIONS properties
        tg_names, tg_opts = udata.get_graph_opts()
        tf_names, tf_opts = udata.get_fn_opts('team')
        nf_names, nf_opts = udata.get_fn_opts('nbhd')
        af_names, af_opts = udata.get_fn_opts('agent')
        
        # Iterate through all points for options
        for tgo, tfo, nfo, afo in zip(
                self.team_graph_opts,
                self.team_fn_opts,
                self.nbhd_fn_opts,
                self.agent_fn_opts
                ):
            
            # Try each team graph option key
            for name in tg_names:
                key = 'team_graph_' + name
                try: tg_opts[key].append(tgo[name])
                except KeyError: tg_opts[key].append('na')
            
            # Try each team function option key
            for name in tf_names:
                key = 'team_fn_' + name
                try: tf_opts[key].append(tfo[name])
                except KeyError: tf_opts[key].append('na')
                
            # Try each team function option key
            for name in nf_names:
                key = 'nbhd_fn_' + name
                try: nf_opts[key].append(nfo[name])
                except KeyError: nf_opts[key].append('na')
                
            # Try each team function option key
            for name in af_names:
                key = 'agent_fn_' + name
                try: af_opts[key].append(afo[name])
                except KeyError: af_opts[key].append('na')
                
        # Write each set of options to points
        dicts_list = [tg_opts, tf_opts, nf_opts, af_opts]
        for dict_ in dicts_list:
            for key, value in dict_.items():
                points[key] = value
        
        # Remove extras
        del_list = ['team_graph_opts', 'team_fn_opts',
                    'nbhd_fn_opts', 'agent_fn_opts']
        for key in del_list:
            del points[key]
        
        return points
        
    def get_dataframe(self):
        
        return pd.DataFrame(self.get_dict())
    
    