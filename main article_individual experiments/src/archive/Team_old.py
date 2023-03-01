# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:42:59 2021

@author: John Meluso
"""

# Import libraries
import networkx as nx
from numpy import array
from numpy.random import default_rng

# Import model files
from Objective import Objective

# Create the random number generator
rng = default_rng()


class Team(nx.Graph):
    
    def __init__(self, team_size, team_graph_type, team_graph_opts, 
                 team_fn_type, team_fn_opts, agent_fn_type, agent_fn_opts,
                 agent_steplim):
        '''Initializes team as a networkx graph with agents as nodes.'''
        
        # Inhert nx.Graph stuff but with additional inputs
        super().__init__()
        
        # Declare fixed indices
        self.ind_ag = 0
        
        # Build network from inputs
        self.build_network(team_graph_type, team_size, team_graph_opts)
        
        # Build agents from inputs
        self.build_agents(agent_fn_type, agent_fn_opts, agent_steplim)
        
        # Declare team objective
        self.build_team(team_fn_type, team_fn_opts)
        
        # Update neighbors
        self.update()
        
    def build_network(self, team_graph_type, team_size, team_graph_opts):
        '''Builds the graph for the team.'''
        
        # Get graph from networkx
        if team_graph_type == 'small_world':
            if team_graph_opts == ():
                team_graph_opts['neighbors'] = 2
                team_graph_opts['prob'] = 0
            graph = nx.watts_strogatz_graph(team_size,
                                            team_graph_opts['neighbors'],
                                            team_graph_opts['prob'])
        elif team_graph_type == 'power':
            if team_graph_opts == {}:
                team_graph_opts['edges'] = 2
                team_graph_opts['prob'] = 0.3
            graph = nx.powerlaw_cluster_graph(team_size,
                                              team_graph_opts['edges'],
                                              team_graph_opts['prob'])
        elif team_graph_type == 'random':
            if team_graph_opts == {}:
                team_graph_opts['prob'] = 0.3
            graph = nx.gnp_random_graph(team_size,
                                        team_graph_opts['prob'])
        elif team_graph_type == 'empty':
            graph = nx.empty_graph(team_size)
        else: # team_graph_type == 'complete' or anything else
            graph = nx.complete_graph(team_size)
        
        # Add graph to self
        self.add_nodes_from(graph)
        self.add_edges_from(graph.edges)
        
    def build_agents(self, agent_fn_type, agent_fn_opts, agent_steplim=1):
        '''Build agents from graph and inputs'''
        
        # Populate nodes with agent properties
        for ag in self:
            self.nodes[ag]['fn'] = Objective(agent_fn_type, agent_fn_opts)
            self.nodes[ag]['x'] = rng.random()
            self.nodes[ag]['fx'] = 1
            self.nodes[ag]['k'] = self.get_ks_agent(ag)
            self.nodes[ag]['agent_steplim'] = agent_steplim
            
    def build_team(self, team_fn_type, team_fn_opts):
        '''Builds team objective from team specifications.'''
        
        # Set objective for team
        self.team_fn_type = Objective(team_fn_type, team_fn_opts)
            
        # Set other team properties
        self.team_k = array([kk for nn, kk in self.degree])
    
    def update(self):
        '''Updates each agent by populating their own value, getting their
        neighbors' current values, and then calculating fx based on them.
        Then, updates the system value.'''
        
        # Cycle through agents
        for ag in self:
            
            # Get current x's and k's
            self.nodes[ag]['x_curr'] = self.get_xs_agent(ag)
            
            # Evaluate by passing x_nbrs and k_nbrs values into x_curr
            self.nodes[ag]['fx'] = self.get_fx_agent(
                ag,self.nodes[ag]['x_curr'],self.nodes[ag]['k'])
        
        # Update system performance
        self.team_performance \
            = self.team_fn_type(array(self.get_fxs_all()), self.team_k)
    
    def step(self):
        '''Step forward in time by syncing all values, and then having each
        agent perform their greedy update.'''
        
        # Calculate a new value for each agent
        for ag in self:
            
            # Get new values
            x_new = self.nodes[ag]['x_curr'].copy()
            x_new[self.ind_ag] = rng.random()
            
            # Update with the better of the two
            if self.get_fx_agent(ag, x_new, self.nodes[ag]['k']) \
                < self.nodes[ag]['fx']:
                self.nodes[ag]['x'] = x_new[self.ind_ag]
            # else keep the current x value
        
        # Update agents' current values
        self.update()
        
    def random(self, ag):
        '''Generates and returns a new random value for the specified agent
        within its allowable step domain. The default max step is 1. The
        range for all agents is [0,1).'''
        x = self.nodes[ag]['x']
        st = self.nodes[ag]['agent_steplim']
        
        # If max step is >= 1, always in domain [0,1)
        if st >= 1:
            return rng.random()
        else:
            
            # If x + max step < 0, use that as upper limit. Else, 1.
            if x + st < 1:
                upper = x + st
            else:
                upper = 1
            
            # If x - max step > 0, use that as lower limit. Else, 0.
            if x - st > 0:
                lower = x - st
            else:
                lower = 0
                
            # Return random value in appropriate range
            return rng.uniform(lower,upper)
        
    def get_xs_agent(self, ag):
        '''Get x's for a specified agent.'''
        x_vect = [self.nodes[nbr]['x'] for nbr in self.neighbors(ag)]
        x_vect.insert(0,self.nodes[ag]['x'])
        return x_vect
    
    def get_ks_agent(self, ag):
        '''Get k's with respect to a specified agent.'''
        k_vect = [self.degree[nbr] for nbr in self.neighbors(ag)]
        k_vect.insert(0,self.degree[ag])
        return k_vect
    
    def get_fx_agent(self, ag, xx, kk):
        '''Calculate f(x) for a specified agent (ag) with xx and kk.'''
        return self.nodes[ag]['fn'](xx,kk)
    
    def get_fxs_all(self):
        '''Get f(x)'s for all agents.'''
        return [self.nodes[ag]['fx'] for ag in self]    
    
    def get_performance(self):
        '''Returns the team performance value.'''
        return self.team_performance
    
        

