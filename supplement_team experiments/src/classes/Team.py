# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 16:42:59 2021

@author: John Meluso
"""

# Import libraries
import networkx as nx
from networkx import classes, NetworkXError
import networkx.algorithms.assortativity as assort
import networkx.algorithms.centrality as cent
import networkx.algorithms.cluster as clust
import networkx.algorithms.shortest_paths.generic as paths
import networkx.algorithms.distance_measures as dist
from numpy import abs, array, float64, fromiter, nan
from numpy.random import default_rng
from scipy.stats import spearmanr as corr
from statistics import mean, stdev
import warnings

# Import model files
import classes.Objective as Objective
import util.graphs as ug

# Create the random number generator
# rng = default_rng(seed=42)
rng = default_rng()
        

class TeamType3xx(nx.Graph):
    
    ### Model Logic Methods ################################################
    
    def __init__(self, team_size, team_graph_type, team_graph_opts, 
                 team_fn_type, team_fn_opts, nbhd_fn_type, nbhd_fn_opts, 
                 agent_fn_type, agent_fn_opts, agent_steplim):
        '''Initializes team as a networkx graph with agents as nodes.'''
        
        # Inhert nx.Graph stuff but with additional inputs
        super().__init__()
        
        # Declare fixed index of the focal agent
        self.ind_ag = 0
        
        # Set number of monte carlo trials to run
        self.mc_trials = 100
        
        # Build network from inputs
        self.build_network(team_graph_type, team_size, team_graph_opts)
        
        # Declare team objective
        self.build_team(team_fn_type, team_fn_opts)
        
        # Build neighborhoods from inputs
        self.build_neighborhoods(nbhd_fn_type, nbhd_fn_opts)
        
        # Build agents from inputs
        self.build_agents(agent_fn_type, agent_fn_opts, agent_steplim)
        
        # Update neighbors
        self.update()
        
        # Calculate network and function metrics
        self.calc_metrics()
        
    def update(self):
        '''Updates each neighborhood's f(x) and the team f(x).'''
        
        # Get old fx
        try: fx_old = self.get_team_fx()
        except: fx_old = nan
        
        # Update neighborhood function for each agent.
        for ag in self:
            self.set_nbhd_fx(ag, self.eval_nbhd_fn(ag, self.get_nbhd_xs(ag)))
        
        # Update system performance
        fx_curr = self.eval_team_fn(self.get_team_xs())
        self.set_team_fx(fx_curr)
        
        # Update system productivity
        try: self.set_team_dfdt((fx_curr-fx_old)/len(self))
        except: self.set_team_dfdt(nan)
        
    def step(self):
        '''Step forward in time by syncing all values, and then having each
        agent perform their greedy update.'''
        
        # Update x and f(x) for each agent
        for ag in self:
            self.set_agent_x_new(ag)
            self.set_agent_fx_new(ag)
            
        # Get old g(x) and new g(x) for each neighborhood
        for ag in self:
            self.set_nbhd_fx(ag,
                 self.eval_nbhd_fn(ag, self.get_nbhd_xs(ag)))
            self.set_nbhd_fx_new(ag,
                 self.eval_nbhd_fn(ag, self.get_nbhd_xs_new(ag)))
        
        # Update if the new x is better
        for ag in self:
            if self.get_nbhd_fx_new(ag) > self.get_nbhd_fx(ag):
                self.set_agent_x(ag, self.get_agent_x_new(ag))
                
        # No matter what, this turn they were as good as fx_new
        for ag in self:
            self.set_agent_fx(ag, self.get_agent_fx_new(ag))
            self.set_nbhd_fx(ag, self.get_nbhd_fx_new(ag))
            
        # Get old and current fx
        fx_old = self.get_team_fx()
        fx_curr = self.eval_team_fn(self.get_team_xs_new())
        
        # Update system performance and productivity
        self.set_team_fx(fx_curr)
        self.set_team_dfdt((fx_curr-fx_old)/len(self))
        
    def random_from_range(self, ag):
        '''Generates and returns a new random value for the specified agent
        within its allowable step domain. The default max step is 1. The
        range for all agents is [0,1).'''
        x = self.get_agent_x(ag)
        st = self.get_agent_steplim(ag)
        
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
    
    
    ### Construction methods ###############################################
        
    def build_network(self, team_graph_type, team_size, team_graph_opts):
        '''Builds the graph for the team.'''
        
        # Get graph from networkx
        graph = get_graph(team_graph_type, team_size, **team_graph_opts)
        
        # Add graph to self
        self.add_nodes_from(graph)
        self.add_edges_from(graph.edges)
        
    def build_team(self, team_fn_type, team_fn_opts):
        '''Builds team objective from team specifications.'''
        
        # Set team properties
        self.set_team_ks()
        
        # Set objective for team
        self.set_team_fn(team_fn_type, team_fn_opts)
    
    def build_neighborhoods(self, nbhd_fn_type, nbhd_fn_opts):
        '''Builds neighborhood objective from neighborhood specifications.'''
        
        # Set neighborhood function for each node
        for ag in self: self.set_nbhd_fn(ag, nbhd_fn_type, nbhd_fn_opts)
        
    def build_agents(self, agent_fn_type, agent_fn_opts, agent_steplim):
        '''Build agents from graph and inputs.'''
        
        # Populate nodes with agent properties
        for ag in self:
            
            # Set agent properties & function
            self.set_agent_steplim(ag, agent_steplim)
            self.set_agent_fn(ag, agent_fn_type, agent_fn_opts)
                
            # Initialize agent x and f(x)
            self.set_agent_x(ag, rng.random())
            self.set_agent_fx(ag, self.eval_agent_fn(ag, self.get_agent_x(ag)))
    
    
    ### Metric Calculators #################################################
        
    def calc_metrics(self):
        '''Calculate the network and function metrics for the team.'''
        self.calc_network_metrics()
        self.calc_fn_metrics()
        
    def calc_network_metrics(self):
        '''Calculate a bunch of different network metrics for the team.'''
        
        # Calculate degree centrality
        tgcd = cent.degree_centrality(self).values()
        self.set_team_graph_centrality_degree_mean(mean(tgcd))
        self.set_team_graph_centrality_degree_stdev(stdev(tgcd))
        
        # Calculate betweenness centrality
        tgcb = cent.betweenness_centrality(self).values()
        self.set_team_graph_centrality_betweenness_mean(mean(tgcb))
        self.set_team_graph_centrality_betweenness_stdev(stdev(tgcb))
        
        # Calculate eigenvector centrality
        try:
            tgce = cent.eigenvector_centrality(
                self,tol=1e-3,max_iter=1000).values()
            self.set_team_graph_centrality_eigenvector_mean(mean(tgce))
            self.set_team_graph_centrality_eigenvector_stdev(stdev(tgce))
        except NetworkXError:
            self.set_team_graph_centrality_eigenvector_mean(nan)
            self.set_team_graph_centrality_eigenvector_stdev(nan)
            
        # Calculate nearest neighbor degree
        tgknn = [knn/(len(self) - 1) for knn
                 in assort.average_neighbor_degree(self).values()]
        self.set_team_graph_nearest_neighbor_degree_mean(mean(tgknn))
        self.set_team_graph_nearest_neighbor_degree_stdev(stdev(tgknn))
        
        # Calculate clustering
        self.set_team_graph_clustering(clust.average_clustering(self))
        
        # Calculate density
        self.set_team_graph_density(classes.function.density(self))
        
        # Calculate assortativity
        warnings.filterwarnings("error")
        try: self.set_team_graph_assortativity(
            assort.degree_assortativity_coefficient(self))
        except ValueError:
            self.set_team_graph_assortativity(nan)
        except RuntimeWarning as rw:
            if rw.args[0] == 'invalid value encountered in double_scalars':
                self.set_team_graph_assortativity(0)
            elif rw.args[0] == 'invalid value encountered in true_divide':
                self.set_team_graph_assortativity(nan)
            else: # unknown error
                raise
        warnings.filterwarnings("default")
        
        # Calculate path length
        try:
            tgpl = paths.average_shortest_path_length(self)
            self.set_team_graph_pathlength(tgpl)
        except NetworkXError:
            self.set_team_graph_pathlength(nan)
         
        # Calculate diameter
        try:
            self.set_team_graph_diameter(dist.diameter(self))
        except NetworkXError:
            self.set_team_graph_diameter(nan)
    
    def calc_fn_metrics(self):
        '''Calculate the appropriate function metrics for each agent,
        neighborhood, and the team.'''
        
        # Generate Monte Carlo samples and calculate integrals while looping
        self.gen_mc_samples()
        
        # Cycle through each pair to calculate alignments & interdependences
        for (ag1, ag2) in self.edges(): self.calc_align_interdep(ag1, ag2)
            
        # Assign results to dictionaries for points
        self.store_fn_metrics()
    
    def gen_mc_samples(self):
        '''Generates random values for Monte Carlo sampling, and calulates f(x)
        for each set of values.'''
        
        ## Generate agent, neighborhood, and team values, respectively
        mc_xlist_agents = self.draw_xs(len(self))
        mc_xlist_nbhds = self.draw_xs(len(self))
        mc_xlist_team = self.draw_xs(len(self))
        
        # Cycle through each agent to calculate f(x)
        for ag in self:
            
            # Calculate agent f(x)'s and integrals while we're at it
            x_list = self.map_agent_xs(ag, mc_xlist_agents)
            fn = lambda xx: self.eval_agent_fn(ag, xx)
            self.set_agent_mc_fx(ag, self.calc_fx(x_list, fn))
            self.set_agent_integral(ag,
                self.calc_integral(self.get_agent_mc_fx(ag)))
            
            # Calculate neighborhood f(x)'s and integrals while we're at it
            x_list = self.map_nbhd_xs(ag, mc_xlist_nbhds)
            fn = lambda xx: self.eval_nbhd_fn(ag, xx)
            self.set_nbhd_mc_fx(ag, self.calc_fx(x_list, fn))
            self.set_nbhd_integral(ag,
                self.calc_integral(self.get_nbhd_mc_fx(ag)))
            
        # Calculate team f(x)
        fn = self.eval_team_fn
        self.set_team_mc_fx(self.calc_fx(mc_xlist_team, fn))
        self.set_team_integral(self.calc_integral(self.get_team_mc_fx()))
        
    def calc_integral(self, y):
        '''Calculates the integral of the function by Monte Carlo integrating
        over the function and subtracting its area from 1'''
        return 1 - y.mean()
    
    def calc_align_interdep(self, ag1, ag2):
        '''Calculates how closely aligned and interdependent the two functions
        are by Monte Carlo integration.'''
        
        # Calculate value for each x and fn
        y1 = self.get_nbhd_mc_fx(ag1)
        y2 = self.get_nbhd_mc_fx(ag2)
        
        # Get correlation of fn outputs
        interdep, __ = corr(y1,y2)
        
        # Get alignment of two functions
        alignment = 1 - mean(abs(y1-y2))
        
        # Save the alignment and interdependence
        self.set_edge_alignment(ag1, ag2, alignment)
        self.set_edge_interdep(ag1, ag2, interdep)
    
    def store_fn_metrics(self):
        '''Writes the function metrics for the team, neighborhoods, and agents
        to the appropriate dictionary structures for returning to Points.'''
        
        # Populate difficulty with integrals
        self.team_fn_diff_integral = self.get_team_integral()
        self.nbhd_fn_diff_integral = self.get_nbhd_integral_mean()
        self.agent_fn_diff_integral = self.get_agent_integral_mean()
        
        # Populate difficulty with peaks
        self.team_fn_diff_peaks = self.get_team_peaks()
        self.nbhd_fn_diff_peaks = self.get_nbhd_peaks_mean()
        self.agent_fn_diff_peaks = self.get_agent_peaks_mean()
        
        # Populate edge metrics
        self.team_fn_alignment = self.get_alignment_mean()
        self.team_fn_interdep = self.get_interdep_mean()
        
    def draw_xs(self, dim, a=0, b=1):
        '''Draw values of x on the domain a to be for each parameter in dim.'''
        return rng.uniform(a, b, (self.mc_trials, dim))
    
    def calc_fx(self, x_list, fn):
        '''Calculate f(x) for each value of x and put into numpy array.'''
        return fromiter([fn(x) for x in x_list], dtype=float64)
    
    
    ### Team Methods #######################################################
    
    def set_team_ks(self):
        '''Set all degrees for the team'''
        self.team_ks = array([kk for nn, kk in self.degree])
    
    def get_team_ks(self):
        '''Get all of the degrees for the team.'''
        return self.team_ks
    
    # No need to set team x's because provided by agents
    
    def get_team_xs(self):
        '''Get all x's for the team from the neighborhood f(x)'s.'''
        return array([self.get_agent_x(ag) for ag in self])
		
	# No need to set team x_new's because provided by agents
	
    def get_team_xs_new(self):
        '''Get all x_new's for the team from the neighborhood f(x)'s.'''
        return array([self.get_agent_x_new(ag) for ag in self])
    
    def set_team_fx(self, val):
        '''Set team f(x) with val.'''
        self.team_performance = val
    
    def get_team_fx(self):
        '''Get team f(x) from the team performance attribute.'''
        return self.team_performance
    
    def set_team_dfdt(self, val):
        '''Set team df(x)/dt with val.'''
        self.team_productivity = val
        
    def get_team_dfdt(self):
        '''Get team df(x)/dt from the team productivity attribute.'''
        return self.team_productivity
    
    def set_team_fn(self, team_fn_type, team_fn_opts):
        '''Set team objective from function type and options.'''
        self.team_fn \
            = get_objective(team_fn_type, team_fn_opts, self.get_team_ks())
    
    def eval_team_fn(self, xx):
        '''Evaluate the team's objective function.'''
        return self.team_fn(xx)
    
    
    ### Neighborhood Methods ###############################################
    
    # No need to set neighborhood k's because provided by NetworkX
    
    def get_nbhd_ks(self, ag):
        '''Get k's with respect to a specified agent.'''
        k_vect = [self.degree[nbr] for nbr in self.neighbors(ag)]
        k_vect.insert(0,self.degree[ag])
        return k_vect
    
    # No need to set neighborhood x's because extracted from neighbors
        
    def get_nbhd_xs(self, ag):
        '''Get neighborhood x's with respect to a specified agent.'''
        x_vect = [self.get_agent_x(nbr) for nbr in self.neighbors(ag)]
        x_vect.insert(0, self.get_agent_x(ag))
        return x_vect
    
    # No need to set neighborhood x_new's because extracted from neighbors
        
    def get_nbhd_xs_new(self, ag):
        '''Get neighborhood x's with respect to a specified agent for all
        agents.'''
        x_vect = [self.get_agent_x_new(nbr) for nbr in self.neighbors(ag)]
        x_vect.insert(0, self.get_agent_x_new(ag))
        return x_vect
    
    def set_nbhd_fx(self, ag, val):
        '''Set neighborhood f(x) with respect to the specified agent & val.'''
        self.nodes[ag]['nbhd_fx'] = val
        
    def get_nbhd_fx(self, ag):
        '''Get neighborhood f(x) with respect to the specified agent.'''
        return self.nodes[ag]['nbhd_fx']
    
    def set_nbhd_fx_new(self, ag, val):
        '''Set new neighborhood f(x) with respect to the specified agent &
        val.'''
        self.nodes[ag]['nbhd_fx_new'] = val
        
    def get_nbhd_fx_new(self, ag):
        '''Get new neighborhood f(x) with respect to the specified agent.'''
        return self.nodes[ag]['nbhd_fx_new']
    
    def set_nbhd_fn(self, ag, nbhd_fn_type, nbhd_fn_opts):
        '''Set neighborhood objective from function type and options.'''
        self.nodes[ag]['nbhd_fn'] \
            = get_objective(nbhd_fn_type, nbhd_fn_opts, self.get_nbhd_ks(ag))
    
    def eval_nbhd_fn(self, ag, xx):
        '''Evaluate the neighborhood objective function with respect to the
        specified agent.'''
        return self.nodes[ag]['nbhd_fn'](xx)
    
    
    ### Agent Methods ######################################################
    
    # No need to set agent k because provided by NetworkX
    
    def get_agent_k(self, ag):
        '''Get k for a specified agent.'''
        return self.degree(ag)
    
    def set_agent_steplim(self, ag, val):
        '''Set the step limit for the specified agent with val.'''
        self.nodes[ag]['agent_steplim'] = val
        
    def get_agent_steplim(self, ag):
        '''Get the step limit for the specified agent.'''
        return self.nodes[ag]['agent_steplim']
    
    def set_agent_x(self, ag, val):
        '''Set x for a specified agent with val.'''
        self.nodes[ag]['agent_x'] = val
    
    def get_agent_x(self, ag):
        '''Get x for a specified agent.'''
        return self.nodes[ag]['agent_x']
    
    def set_agent_fx(self, ag, val):
        '''Set f(x) for a specified agent with val.'''
        self.nodes[ag]['agent_fx'] = val
    
    def get_agent_fx(self, ag):
        '''Calculate f(x) for a specified agent (ag) with xx and kk.'''
        return self.nodes[ag]['agent_fx']
    
    def set_agent_x_new(self, ag):
        '''Set the potential new x for a specified agent from fn random.'''
        self.nodes[ag]['agent_x_new'] = self.random_from_range(ag)
    
    def get_agent_x_new(self, ag):
        '''Get the potential new x for a specified agent.'''
        return self.nodes[ag]['agent_x_new']
    
    def set_agent_fx_new(self, ag):
        '''Set the potential new f(x) for a specified agent with the evaluated
        agent x_new.'''
        self.nodes[ag]['agent_fx_new'] \
            = self.eval_agent_fn(ag, self.get_agent_x_new(ag))
    
    def get_agent_fx_new(self, ag):
        '''Get the potential new f(x) for a specified agent.'''
        return self.nodes[ag]['agent_fx_new']
    
    def set_agent_fn(self, ag, agent_fn_type, agent_fn_opts):
        '''Set agent objective from function type and options.'''
        self.nodes[ag]['agent_fn'] \
            = get_objective(agent_fn_type, agent_fn_opts, self.get_agent_k(ag))
    
    def eval_agent_fn(self, ag, xx):
        '''Evaluate the specified agent's objective function.'''
        return self.nodes[ag]['agent_fn'](xx)
    
    
    ### Edge Methods #######################################################
    
    def set_edge_alignment(self, ag1, ag2, val):
        '''Sets the alignment of the two agents 1 and 2 to val.'''
        self.edges[ag1, ag2]['alignment'] = val
        
    def get_edge_alignment(self, ag1, ag2):
        '''Gets the alignment of agents 1 and 2.'''
        return self.edges[ag1, ag2]['alignment']
        
    def set_edge_interdep(self, ag1, ag2, val):
        '''Sets the interdependence of the two agents 1 and 2 to val.'''
        self.edges[ag1, ag2]['interdep'] = val
        
    def get_edge_interdep(self, ag1, ag2):
        '''Gets the interdependence of the two agents 1 and 2.'''
        return self.edges[ag1, ag2]['interdep']
    
    
    ### Network Metric Setters & Getters ###################################
        
    def set_team_graph_centrality_degree_mean(self, val):
        '''Sets the team's degree centrality mean.'''
        self.team_graph_centrality_degree_mean = val
        
    def get_team_graph_centrality_degree_mean(self):
        '''Gets the team's degree centrality mean.'''
        return self.team_graph_centrality_degree_mean
    
    def set_team_graph_centrality_degree_stdev(self, val):
        '''Sets the team's degree centrality standard deviation.'''
        self.team_graph_centrality_degree_stdev = val
        
    def get_team_graph_centrality_degree_stdev(self):
        '''Gets the team's degree centrality standard deviation.'''
        return self.team_graph_centrality_degree_stdev
    
    def set_team_graph_centrality_eigenvector_mean(self, val):
        '''Sets the team's eigenvector centrality mean.'''
        self.team_graph_centrality_eigenvector_mean = val
        
    def get_team_graph_centrality_eigenvector_mean(self):
        '''Gets the team's eigenvector centrality mean.'''
        return self.team_graph_centrality_eigenvector_mean
    
    def set_team_graph_centrality_eigenvector_stdev(self, val):
        '''Sets the team's eigenvector centrality standard deviation.'''
        self.team_graph_centrality_eigenvector_stdev = val
        
    def get_team_graph_centrality_eigenvector_stdev(self):
        '''Gets the team's eigenvector centrality standard deviation.'''
        return self.team_graph_centrality_eigenvector_stdev
    
    def set_team_graph_centrality_betweenness_mean(self, val):
        '''Sets the team's betweenness centrality mean.'''
        self.team_graph_centrality_betweenness_mean = val
        
    def get_team_graph_centrality_betweenness_mean(self):
        '''Gets the team's betweenness centrality mean.'''
        return self.team_graph_centrality_betweenness_mean
    
    def set_team_graph_centrality_betweenness_stdev(self, val):
        '''Sets the team's betweenness centrality standard deviation.'''
        self.team_graph_centrality_betweenness_stdev = val
        
    def get_team_graph_centrality_betweenness_stdev(self):
        '''Gets the team's betweenness centrality standard deviation.'''
        return self.team_graph_centrality_betweenness_stdev
    
    def set_team_graph_nearest_neighbor_degree_mean(self, val):
        '''Sets the team's nearest neighbor degree mean.'''
        self.team_graph_nearest_neighbor_degree_mean = val
        
    def get_team_graph_nearest_neighbor_degree_mean(self):
        '''Gets the team's nearest neighbor degree mean.'''
        return self.team_graph_nearest_neighbor_degree_mean
    
    def set_team_graph_nearest_neighbor_degree_stdev(self, val):
        '''Sets the team's nearest neighbor degree standard deviation.'''
        self.team_graph_nearest_neighbor_degree_stdev = val
        
    def get_team_graph_nearest_neighbor_degree_stdev(self):
        '''Gets the team's nearest neighbor degree standard deviation.'''
        return self.team_graph_nearest_neighbor_degree_stdev
    
    def set_team_graph_clustering(self, val):
        '''Sets the clustering of the team.'''
        self.team_graph_clustering = val
        
    def get_team_graph_clustering(self):
        '''Gets the clustering of the team.'''
        return self.team_graph_clustering
    
    def set_team_graph_density(self, val):
        '''Sets the density of the team.'''
        self.team_graph_density = val
        
    def get_team_graph_density(self):
        '''Gets the density of the team.'''
        return self.team_graph_density
    
    def set_team_graph_assortativity(self, val):
        '''Sets the assortativity of the team.'''
        self.team_graph_assortativity = val
        
    def get_team_graph_assortativity(self):
        '''Gets the assortativity of the team.'''
        return self.team_graph_assortativity
    
    def set_team_graph_pathlength(self, val):
        '''Sets the pathlength of the team.'''
        self.team_graph_pathlength = val
        
    def get_team_graph_pathlength(self):
        '''Gets the pathlength of the team.'''
        return self.team_graph_pathlength
    
    def set_team_graph_diameter(self, val):
        '''Sets the diameter of the team.'''
        self.team_graph_diameter = val
        
    def get_team_graph_diameter(self):
        '''Gets the diameter of the team.'''
        return self.team_graph_diameter
    
    
    ### Function Metric Setters & Getters ##################################
    
    ## Team metrics
    
    def set_team_mc_fx(self, val):
        '''Sets the f(x) team value from Monte Carlo sampling to val.'''
        self.team_mc_fx = val
        
    def get_team_mc_fx(self):
        '''Gets the f(x) team value from Monte Carlo sampling.'''
        return self.team_mc_fx
    
    def set_team_integral(self, val):
        '''Sets the integral for the team with val.'''
        self.team_integral = val
        
    def get_team_integral(self):
        '''Gets the integral for the team.'''
        return self.team_integral
    
    def get_team_fn_diff_integral(self):
        '''Get team integral.'''
        return self.team_fn_diff_integral
    
    def get_team_peaks(self):
        '''Get the number of peaks for the team function.'''
        return self.team_fn.get_peaks()
    
    def get_team_fn_diff_peaks(self):
        '''Get the numbef of peaks metric for the team function.'''
        return self.team_fn_diff_peaks
    
    def get_team_fn_alignment(self):
        '''Get team alignment.'''
        return self.team_fn_alignment
    
    def get_team_fn_interdep(self):
        '''Get team interdependence.'''
        return self.team_fn_interdep
    
    ## Neighborhood metrics
    
    def set_nbhd_mc_fx(self, ag, val):
        '''Sets the f(x) value for neighborhood from Monte Carlo to val.'''
        self.nodes[ag]['nbhd_mc_fx'] = val
        
    def get_nbhd_mc_fx(self, ag):
        '''Gets the f(x) value for neighborhood from Monte Carlo sampling.'''
        return self.nodes[ag]['nbhd_mc_fx']
    
    def map_nbhd_xs(self, ag, x_all):
        '''Get the x's corresponding to agent and agent's neighbors from
        x_list from the indices on the specified axis.'''
        x_vect = [x_all[:,nb] for nb in self.neighbors(ag)]
        x_vect.insert(self.ind_ag,x_all[:,ag])
        return array(x_vect).transpose()
    
    def set_nbhd_integral(self, ag, val):
        '''Sets the integral for the neighborhood of agent with val.'''
        self.nodes[ag]['nbhd_integral'] = val
    
    def get_nbhd_integral(self, ag):
        '''Gets the integral for the neighborhood of agent.'''
        return self.nodes[ag]['nbhd_integral']
    
    def get_nbhd_integral_mean(self):
        '''Gets the mean integral across all neighborhoods.'''
        return mean([self.get_nbhd_integral(ag) for ag in self])
    
    def get_nbhd_fn_diff_integral(self):
        '''Get neighborhood integral.'''
        return self.nbhd_fn_diff_integral
    
    def get_nbhd_peaks(self, ag):
        '''Gets the number of peaks for the neighborhood of agent.'''
        return self.nodes[ag]['nbhd_fn'].get_peaks()
    
    def get_nbhd_peaks_mean(self):
        '''Gets the mean number of peaks across all neighborhoods.'''
        return mean([self.get_nbhd_peaks(ag) for ag in self])
    
    def get_nbhd_fn_diff_peaks(self):
        '''Get neighborhood integral.'''
        return self.nbhd_fn_diff_peaks
    
    ## Agent metrics
    
    def set_agent_mc_fx(self, ag, val):
        '''Sets the f(x) value for agent from Monte Carlo sampling to val.'''
        self.nodes[ag]['agent_mc_fx'] = val
        
    def get_agent_mc_fx(self, ag):
        '''Gets the f(x) value for agent from Monte Carlo sampling.'''
        return self.nodes[ag]['agent_mc_fx']
    
    def map_agent_xs(self, ag, x_all):
        '''Get the x's corresponding to agent from x_list from the index
        on the specified axis.'''
        return array(x_all[:,ag]).transpose()
    
    def set_agent_integral(self, ag, val):
        '''Sets the integral for the agent to val.'''
        self.nodes[ag]['agent_integral'] = val
        
    def get_agent_integral(self, ag):
        '''Gets the integral for the agent.'''
        return self.nodes[ag]['agent_integral']
    
    def get_agent_integral_mean(self):
        '''Gets the mean integral across all agents.'''
        return mean([self.get_agent_integral(ag) for ag in self])
        
    def get_agent_fn_diff_integral(self):
        '''Get agent integral.'''
        return self.agent_fn_diff_integral
    
    def get_agent_peaks(self, ag):
        '''Get the number of peaks for the agent.'''
        return self.nodes[ag]['agent_fn'].get_peaks()
    
    def get_agent_peaks_mean(self):
        '''Gets the mean number of peaks across all agents.'''
        return mean([self.get_agent_peaks(ag) for ag in self])
    
    def get_agent_fn_diff_peaks(self):
        '''Get agent peaks.'''
        return self.agent_fn_diff_peaks
    
    ## Edge metrics
    
    def get_interdep_mean(self):
        '''Gets the mean interdep across all edges.'''
        if len(self.edges()) > 0:
            return mean([self.get_edge_interdep(ag1, ag2) for (ag1, ag2) \
                         in self.edges()])
        else: return 0
    
    def get_alignment_mean(self):
        '''Gets the mean alignment across all edges.'''
        if len(self.edges()) > 0:
            return mean([self.get_edge_alignment(ag1, ag2) for (ag1, ag2) \
                         in self.edges()])
        else: return 0
    

class TeamType3fx(TeamType3xx):
    
    ### Neighborhood Methods ###############################################
        
    def get_nbhd_xs(self, ag):
        '''Get neighborhood x's with respect to a specified agent.'''
        x_vect = [self.get_agent_fx(nbr) for nbr in self.neighbors(ag)]
        x_vect.insert(0, self.get_agent_fx(ag))
        return x_vect
    
    def get_nbhd_xs_new(self, ag):
        '''Get neighborhood x's with respect to a specified agent, and just
        the focal agent's new x.'''
        x_vect = [self.get_agent_fx(nbr) for nbr in self.neighbors(ag)]
        x_vect.insert(0, self.get_agent_fx_new(ag))
        return x_vect
    

class TeamType3xg(TeamType3xx):
    
    ### Team Methods #######################################################
    
    def get_team_xs(self):
        '''Get all x's for the team from the neighborhood f(x)'s.'''
        return array([self.get_nbhd_fx(ag) for ag in self])
    
    
class TeamType3fg(TeamType3fx, TeamType3xg):
        
    pass
    
    
def get_objective(fn_type,fn_opts,degrees):
    '''Selects and returns a callable object of the correct function type.'''
    
    try:
        return getattr(Objective,fn_type.capitalize())(fn_opts,degrees)
    except:
        raise RuntimeError(f'Function type {fn_type} is not valid.')
        

def get_graph(graph_type, team_size, **team_graph_opts):
    '''Selects and returns a graph object of the correct type.'''
    
    try:
        return getattr(ug,f'get_{graph_type}')(team_size, **team_graph_opts)
    except:
        raise RuntimeError(f'Graph type {graph_type} is not valid.')
        