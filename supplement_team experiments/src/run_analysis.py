# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:38:02 2022

@author: John Meluso
"""

# import source files
import analysis_regressions as reg
import analysis_random_forest as rf


if __name__ == '__main__':
    
    # Select execution set
    execset = 1
    model = '3xx'
    
    # Regression analyses
    reg.run_all(execset, model)
    
    # Random forest analysis
    rf.run_all(execset, model)