# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:38:02 2022

@author: John Meluso
"""

# import settings file
import fig_settings as fs

# import main doc files
import plot_example_runs as per
import plot_outcome_means as pom
import plot_importances_tasks as pit
import plot_importances_networks as pin

# import supplement files
import plot_joint_grids_vs_baseline as pjg
import plot_random_forest_importances as prfi
import plot_correlation as pc
import plot_regression_coefficients as prc


if __name__ == '__main__':

    # Set dataset parameters
    execset = 1
    model = '3xx'
    base_graph = 'empty'
    steps = [
        0.001,
        0.01,
        0.1,
        1.0
        ]
    size = 9

    #%% SI Appendix plots
    
    # Reset fig settings
    fs.set_fonts()

    # # Example run averages
    # __ = per.plot_example_run_averages(execset)
    
    # # Generate heatmaps
    # for step in steps:
    #     pjg.plot_joint_grid(execset, model, base_graph, step)

    # # Generate relative performances for all graphs
    # pom.plot_page()

    # # Directional importances
    pit.plot_importances_tasks()
    # pin.plot_importances_networks()

    # # Plot random forest importances
    # prfi.plot_randfor_imps()
    
    # # Plot correlations
    # pc.plot_correlations(execset)
