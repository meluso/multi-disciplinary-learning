# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 19:38:08 2021

@author: John Meluso
"""

# Import libraries
from numpy import array, cos, exp, float64, median, pi, prod, sin, sqrt

# Import model classes
from classes.FnOptions import NodeWeight, DegreeWeight
    

class Objective(object):
    
    def __init__(self, opts, kk):
        '''Initialize an objective function with the specified options.'''
        
        # Get the correct weighting function for the specified objective
        self.set_weight(opts)
        self.set_degree(kk)
        self.set_peaks()
        
    def __call__(self, xx):
        '''Call the objective, just getting the main variables for the parent
        class.'''
        self.set_main_vars(xx)
        
    def set_main_vars(self, xx):
        '''Sets the main variables used in most objectives, including the x
        inputs (xx) and degree of each node (kk).'''
        
        self.xx = array(xx) # Convert list to numpy array
        self.nn = self.xx.size
        
    def set_weight(self, opts, divisor=1):
        '''For functions that involve a weight term, set the normalization
        weight from the specified input option.'''
        
        if opts['weight'] == 'node':
            self.norm_weight = NodeWeight(divisor)
        elif opts['weight'] == 'degree':
            self.norm_weight = DegreeWeight(divisor)
        else: raise RuntimeError('Weight ' + opts['weight'] \
                                 + ' for ' + self.__class__.__name__ \
                                 + ' function is invalid.')
            
    def set_degree(self, kk):
        '''Sets the degree of each node (kk).'''
        self.kk = array(kk)
        
    def set_frequency(self, opts):
        '''For functions that involve a frequency term, set the frequency
        from the specified input option.'''
        self.frequency = opts['frequency']
        
    def set_peaks(self, num_peaks=1):
        '''Sets the number of peaks that the function has. Defaults to 1.'''
        self.num_peaks = num_peaks
        
    def calc_peaks(self):
        '''Calculates the number of peaks for functions with more than one.'''
        
        # Build peaks from frequency (because otherwise mostly have 1)
        self.get_frequency()
        
        # If frequency is uniform, every dimension adds peaks w/ same omega
        if self.frequency == 'uniform':
            peaks = float64(self.omega/pi + (1/2))**(len(self.kk))
            
        # If frequency is degree-based, every dimension has diff num of peaks
        elif self.frequency == 'degree':
            peaks = prod(float64(self.kk + 1))
        
        return peaks
    
    def get_peaks(self):
        '''Returns the number of peaks the function has.'''
        return self.num_peaks
        
    def get_frequency(self):
        '''For functions that involve a frequency term, get the frequency.'''
        
        # Set the frequency for the function
        if not(hasattr(self, 'omega')):
            if self.frequency == 'uniform':
                self.omega = 7*pi/2
            elif self.frequency == 'degree':
                self.omega = (1 + 2*self.kk)*pi/2
            else: raise RuntimeError('Frequency ' + self.frequency \
                                     + ' for ' + self.__class__.__name__ \
                                     + ' function is invalid.')
                
    def set_k_exponent(self,opts):
        '''For functions that vary the role of degree k in the exponent, set
        the exponent from the specified input option.'''
        self.exponent = opts['exponent']
        
    def get_k_exponent(self):
        '''For functions that vary the role of degree k in the exponent, get
        the exponent.'''
        
        # Set the exponent for the function
        if not(hasattr(self, 'kk_term')):
            if self.exponent == 'uniform':
                
                # Try indexing for length greater than 1
                try: self.kk_term = self.kk[0]
                except IndexError: self.kk_term = self.kk
                
            elif self.exponent == 'degree':
                self.kk_term = self.kk
            else: raise RuntimeError('Exponent ' + self.exponent \
                                     + ' for ' + self.__class__.__name__ \
                                     + ' function is invalid.')
        
    def norm_sum(self, xx):
        '''Normalizes & sums a vector of values appropriately by weighting.'''
        return self.norm_weight(xx, self.nn, self.kk)
    

class Average(Objective):
    '''The absolute sum function, a simple sum on the domain x=[0,1].
    This function is aligned.'''
    
    def __init__(self, opts, kk):
        super().__init__(opts, kk)
        
    def __call__(self, xx):
        super().__call__(xx)
        return self.norm_sum(self.xx)


class Sphere(Objective):
    '''The sphere function, a sum of squares on the domain x=[0,1].
    This function is aligned.'''
    
    def __init__(self, opts, kk):
        super().__init__(opts, kk)
        
    def __call__(self, xx):
        super().__call__(xx)
        return self.norm_sum(self.xx**2)
    

class Root(Objective):
    '''The square root function, a sum of squares roots on the domain
    x=[0,1]. This function is aligned.'''
    
    def __init__(self, opts, kk):
        super().__init__(opts, kk)
        
    def __call__(self, xx):
        super().__call__(xx)
        return self.norm_sum(sqrt(self.xx))


class Sin2(Objective):
    '''The sine squared function, a sum of squared sines on the domain
    x=[0,1]. This function is aligned for all weights but only with UNIFORM
    frequency.'''
    
    def __init__(self, opts, kk):
        super().__init__(opts, kk)
        
        # Set the frequency for the function
        self.set_frequency(opts)
        
        # Set number of peaks
        self.set_peaks(self.calc_peaks())
        
    def __call__(self, xx):
        super().__call__(xx)
        self.get_frequency()
        return self.norm_sum(sin(self.omega*self.xx)**2)
    

class Sin2sphere(Objective):
    '''The sine squared sphere function, a sum of squared sines and the sphere
    function on the domain x=[0,1]. This function is aligned for all weights
    but only with UNIFORM frequency.'''
    
    def __init__(self, opts, kk):
        
        # Get the correct weighting function for the specified objective
        self.set_weight(opts, divisor=2)
        
        # Set the degree
        self.set_degree(kk)
        
        # Set the frequency for the function
        self.set_frequency(opts)
        
        # Set number of peaks
        self.set_peaks(self.calc_peaks())
        
    def __call__(self, xx):
        super().__call__(xx)
        self.get_frequency()
        return self.norm_sum(sin(self.omega*self.xx)**2 + self.xx**2)
    

class Sin2root(Objective):
    '''The sine squared root function, a sum of squared sines and the root
    function on the domain x=[0,1]. This function is aligned for all weights
    but only with UNIFORM frequency.'''
    
    def __init__(self, opts, kk):
        
        # Get the correct weighting function for the specified objective
        self.set_weight(opts, divisor=2)
        
        # Set the degree
        self.set_degree(kk)
        
        # Set the frequency for the function
        self.set_frequency(opts)
        
        # Set number of peaks
        self.set_peaks(self.calc_peaks())
        
    def __call__(self, xx):
        super().__call__(xx)
        self.get_frequency()
        return self.norm_sum(sin(self.omega*self.xx)**2 + sqrt(self.xx))


class Losqr_hiroot(Objective):
    '''A function which takes the form of a square when the degree of a node
    is 1 and increases up to a square root as a function of degree up to k_tot,
    the total degree of the nodes in the local neighborhood. With x=[0,1].
    This function is not aligned.'''
    
    def __init__(self, opts, kk):
        super().__init__(opts, kk)
        self.set_k_exponent(opts)
        
    def __call__(self, xx):
        super().__call__(xx)
        
        # Try sum for size of greater than 1
        try: ktot = sum(self.kk)
        except TypeError: ktot = self.kk
        
        # Try getting zeroth term for size of greater than 1
        self.get_k_exponent()
        
        # Calculate function
        exponent = 2**(1-((2*self.kk_term)/(ktot + 1)))
        return self.norm_sum(self.xx**exponent)
    
    
class Hisqr_loroot(Objective):
    '''A function which takes the form of a square root when the degree of a
    node is k_tot, the total degree of the nodes in the local neighborhood, and
    decreases down to a square as a function of degree. With x=[0,1]. This
    function is not aligned.'''
    
    def __init__(self, opts, kk):
        super().__init__(opts, kk)
        self.set_k_exponent(opts)
        
    def __call__(self, xx):
        super().__call__(xx)
        
        # Try sum for size of greater than 1
        try: ktot = sum(self.kk)
        except TypeError: ktot = self.kk
        
        # Try getting zeroth term for size of greater than 1
        self.get_k_exponent()
        
        # Calculate function
        exponent = 2**(((2*self.kk_term)/(ktot + 1))-1)
        return self.norm_sum(self.xx**exponent)


class Max(Objective):
    '''The maximum of a vector of values xx, with each x in xx on [0,1]. This
    function is not aligned.'''
    
    def __init__(self, opts, kk):
        
        # Set the degree
        self.set_degree(kk)
        
        # Set number of peaks
        self.set_peaks()
        
    def __call__(self, xx):
        try: return max(xx)
        except TypeError: return xx
    

class Min(Objective):
    '''The minimum of a vector of values xx, with each x in xx on [0,1]. This
    function is not aligned.'''
    
    def __init__(self, opts, kk):
    
        # Set the degree
        self.set_degree(kk)
        
        # Set number of peaks
        self.set_peaks()
        
    def __call__(self, xx):
        try: return min(xx)
        except TypeError: return xx
        
        
class Median(Objective):
    '''The median of a vector of values xx, with each x in xx on [0,1]. This
    function is not aligned.'''
    
    def __init__(self, opts, kk):
        
        # Set the degree
        self.set_degree(kk)
        
        # Set number of peaks
        self.set_peaks()
        
    def __call__(self, xx):
        try: return median(xx)
        except TypeError: return xx
    

class Kth_power(Objective):
    '''Takes each term of a vector xx to the power of its degree, x=[0,1]. This
    function is aligned.'''
    
    def __init__(self, opts, kk):
        super().__init__(opts, kk)
        
    def __call__(self, xx):
        super().__call__(xx)
        return self.norm_sum(self.xx**(self.kk+1))


class Kth_root(Objective):
    '''Takes each term of a vector xx to the root of its degree + 1 to avoid
    undefined errors, with x=[0,1]. This function is aligned.'''
    
    def __init__(self, opts, kk):
        super().__init__(opts, kk)
        
    def __call__(self, xx):
        super().__call__(xx)
        return self.norm_sum(self.xx**(1/(self.kk+1)))
            

class Ackley(Objective):
    '''The Ackley function with constants defined in init and each x=[0,1].
    This function is not aligned.'''
    
    def __init__(self, opts, kk):
        
        # Define Ackley constants (can be set via opts if needed)
        self.aa = 20
        self.bb = 0.2
        self.cc = 7*pi
        
        # Set the degree
        self.set_degree(kk)
        
        # Set number of peaks
        self.set_peaks(4**(len(kk)))
        
    def __call__(self, xx):
        super().__call__(xx)
        
        # Flip x
        self.xx = 1 - self.xx
        
        # Try sums for size of xx greater than 1
        try: xx_term = sum(self.xx**2)
        except TypeError: xx_term = self.xx**2
        try: cos_term = sum(cos(self.cc*self.xx))
        except TypeError: cos_term = cos(self.cc*self.xx)
        
        # Construct function
        term1 = -self.aa*exp(-self.bb*sqrt((1/self.nn)*xx_term))
        term2 = -exp((1/self.nn)*cos_term)
        
        # Define normalizer
        norm = (self.aa*(1-exp(-self.bb)) + (exp(1) - exp(-1)))
        
        # Calculate result
        return 1 - ((term1 + term2 + self.aa + exp(1))/norm)
        

        
if __name__ == '__main__':
    
    from FnOptions import NodeWeight, DegreeWeight
    
    kk = [3, 5]
    opts = {'weight': 'degree', 'frequency':'degree'}
    var = Sin2(opts, kk)
    xx = [0.1, 0.2]
    print(f'Num Peaks = {var.get_peaks()}')
    print(f'Test Eval = {var(xx)}')
    
    kk = [3, 5, 10]
    opts = {'weight': 'node', 'frequency':'uniform'}
    var = Sin2(opts, kk)
    xx = [0.1, 0.2, 0.3]
    print(f'Num Peaks = {var.get_peaks()}')
    print(f'Test Eval = {var(xx)}')