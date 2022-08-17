# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:21:33 2021

@author: John Meluso
"""

# Import libraries
from numpy.random import default_rng
import pandas as pd

# Import model files
import util.params as up

# Create the random number generator
rng = default_rng()


def test_parameters(tests):
    
    cases = []
    
    for test in tests:
        if type(test) == str and test.startswith('Test'):
            pt_count, pt_cases = up.count_and_get_params(
                test,
                all_cases=True
                )
            cases = pd.DataFrame(pt_cases)
        else:
            pt_count, pt_cases = up.count_and_get_params(
                test,
                get_cases=rng.integers(0,pt_count,size=2)
                )
        print('Cases in ' + str(test) + ': ' + str(pt_count))
        
    return cases


if __name__ == '__main__':
    
    # List out tests
    tests = [
        'TestOnce',
        # 'TestModels',
        # 41, 42, 43
        ]
    
    cases = test_parameters(tests)
    
    