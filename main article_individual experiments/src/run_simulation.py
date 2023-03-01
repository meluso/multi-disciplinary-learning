# -*- coding: utf-8 -*-
'''
Created on Wed Nov  3 15:54:50 2021

@author: John Meluso
'''

# Import libraries
import sys
import datetime as dt
import numpy as np
from numpy.random import default_rng

# Import model classes
from classes.Points import Points

# Import model methods
import run_Team as rt
import util.data as ud
import util.params as up

# Create the random number generator
rng = default_rng()


def run_simulation(mode='test',paramset='TestModels'):

    # Get start time
    t_start = dt.datetime.now()

    '''Set running conditions based on platform.'''

    if sys.platform.startswith('linux'):

        # get the number of this job and the total number of jobs from the
        # queue system. These arguments are given by the VACC to this script
        # via submit_job.sh. If there are n jobs to be run total (numruns = n),
        # then casenum should run from 0 to n-1. In notation: [0,n) or [0,n-1].
        try:

            # Get directory and execution number
            outputdir = str(sys.argv[1])
            paramset = int(sys.argv[2])

            # Get specific case-run combo if mode is single, else get run input
            if mode == 'single':
                casenum = int(sys.argv[3])
                subset = 0
            elif mode == 'test':
                print(str(sys.argv[3]) + ' passed to run_simulation.')
                casenum = rng.integers(up.count_params(paramset))
                subset = 0
            elif mode == 'subset':
                subset = int(sys.argv[3])
                numsets = int(sys.argv[4])
            elif mode == 'all':
                subset = 0
            else:
                raise RuntimeError('Mode ' + str(mode) + ' is not valid.')

        except IndexError:
            sys.exit('Usage: %s outputdir paramset subset' % sys.argv[0] )

    else:

        # Get directory and execution number
        outputdir = '../data/exec000'

        # Get specific case-run combo if mode is single, else set top runnum
        if mode == 'single' or mode == 'test':
            casenum = rng.integers(up.count_params(paramset))
            subset = 0
        elif mode == 'subset':
            numsets = 500
            print('Subset simulation with ' + str(numsets) + ' subsets.')
            subset = rng.integers(numsets)
        elif mode == 'all':
            subset = 0
        else:
            raise RuntimeError('Mode ' + str(mode) + ' is not valid.')
            

    '''Run simulation.'''
    
    # Create data object
    points = Points()

    if mode == 'test' or mode == 'single':

        # Get the random case for testing
        case = up.get_params(paramset,get_cases=[casenum])

        # Run simulation for specified set of parameters
        points = rt.run_Team(points,case)

    elif mode == 'subset':

        # Get number of parameter combinations in subset
        par_per_sub = int(np.ceil(up.count_params(paramset)/numsets))

        # Get subset of parameters
        sub_params = up.get_params(
            paramset,
            get_cases=range(par_per_sub*subset,par_per_sub*(subset+1))
            )

        # Loop through all cases
        for case in sub_params:

            # Run simulation for specified set of parameters
            points = rt.run_Team(points,case)

    elif mode == 'all':

        # Loop through all cases
        for case in up.get_params(paramset, all_cases=True):

            # Run simulation for specified set of parameters
            points = rt.run_Team(points,case)

    else:

        raise RuntimeError('Not a valid input. No simulation run.')
    
    # Build name for specific test
    if isinstance(paramset,str): paramset = 0
    pset_str = f'exec{paramset:03}'
    sub_str = f'sub{subset:04}'
    filename = outputdir + '/' + pset_str + '_' + sub_str

    # Save results to location specified by platform
    ud.save_csv(filename,points)

    # Print filename
    print('Output Filename Base: ' + filename)

    # Print end time
    t_stop = dt.datetime.now()

    print('Simulation Time Elapsed: ' + str(t_stop - t_start) + '\n')


if __name__ == '__main__':
    # run_simulation('test')
    # time.sleep(2)
    # run_simulation('single')
    # time.sleep(2)
    run_simulation('subset')
    # time.sleep(2)
    # run_simulation('all')