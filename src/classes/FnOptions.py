# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 19:38:08 2021

@author: John Meluso
"""

# Import libraries
from numpy import dot


class Weight(object):
    
    def __init__(self, divisor):
        '''Initialize function class that normalizes an input by the node
        count and a range equivalent to the divisor input.'''
        self.divisor = divisor


class NodeWeight(Weight):
    
    def __init__(self, divisor=1):
        '''Initialize function class that normalizes an input by the node
        count and a range equivalent to the divisor input.'''
        super().__init__(divisor)
        
    def __call__(self, xx, nn, kk):
        '''Return a normalized function call for the vector xx of length nn.'''
        
        # Try summing xx for when length is greater than 1
        try: xx_term = sum(xx)
        except TypeError: xx_term = xx
        
        # Calculate normalization
        return xx_term/(self.divisor*nn)
    

class DegreeWeight(Weight):
    
    def __init__(self, divisor=1):
        '''Initialize function class that normalizes an input by the degree of
        the node +1 (for each node's self-edge) and a range equivalent to the
        divisor input.'''
        super().__init__(divisor)
        
    def __call__(self, xx, nn, kk):
        '''Return a normalized function call for the vector xx of length nn.'''
        
        # Try summing kk for when length is greater than 1
        try: kk_term = sum(kk + 1)
        except TypeError: kk_term = kk + 1
        
        # Calculate normalization
        return dot(kk + 1, xx)/(self.divisor*kk_term)