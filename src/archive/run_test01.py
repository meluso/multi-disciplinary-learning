# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 23:42:13 2021

@author: John Meluso
"""

import model_team as mt
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    
    #%% Run simulation
    net_type = ['small-world', 'complete', 'power', 'random']
    net_size = [10]
    ag_fn = ['root','absolute-sum','sphere']
    steps = 25
    runs = 100
    results = np.zeros(
        (len(net_type),len(net_size),len(ag_fn),runs,steps + 1))
    
    for ii, nt in enumerate(net_type):
        for jj, ns in enumerate(net_size):
            for kk, af in enumerate(ag_fn):
                for rr in np.arange(runs):
                    team = mt.Team(nt,ns,af)
                    for step in np.arange(steps):
                        team.step()
                    results[ii,jj,kk,rr,:] \
                        = np.average(np.array(team.history.copy()), axis=1)
                    
    #%% Consolidate results
    small_root = np.average(results[0,0,0,:,:], axis=0)
    small_abs = np.average(results[0,0,1,:,:], axis=0)
    small_sphere = np.average(results[0,0,2,:,:], axis=0)
    complete_root = np.average(results[1,0,0,:,:], axis=0)
    complete_abs = np.average(results[1,0,1,:,:], axis=0)
    complete_sphere = np.average(results[1,0,2,:,:], axis=0)
    power_root = np.average(results[2,0,0,:,:], axis=0)
    power_abs = np.average(results[2,0,1,:,:], axis=0)
    power_sphere = np.average(results[2,0,2,:,:], axis=0)
    random_root = np.average(results[2,0,0,:,:], axis=0)
    random_abs = np.average(results[2,0,1,:,:], axis=0)
    random_sphere = np.average(results[2,0,2,:,:], axis=0)
    
    
    #%% Plot results
    plt.plot(small_root,label='small_root')
    plt.plot(small_abs,label='small_abs')
    plt.plot(small_sphere,label='small_sphere')
    plt.plot(complete_root,label='complete_root')
    plt.plot(complete_abs,label='complete_abs')
    plt.plot(complete_sphere,label='complete_sphere')
    plt.plot(power_root,label='power_root')
    plt.plot(power_abs,label='power_abs')
    plt.plot(power_sphere,label='power_sphere')
    plt.plot(random_root,label='random_root')
    plt.plot(random_abs,label='random_abs')
    plt.plot(random_sphere,label='random_sphere')
    plt.legend()
    plt.xlabel('turn number')
    plt.ylabel('average agent f(x)')
    plt.show()
