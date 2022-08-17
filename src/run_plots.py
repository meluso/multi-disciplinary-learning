# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:38:02 2022

@author: John Meluso
"""

# import settings file
import fig_settings as fs

# import main doc files
import plot_tasks_networks_difficulties as ptnd
import plot_example_runs as per
import plot_outcome_means as pom
import plot_importances_tasks as pit
import plot_importances_networks as pin
import plot_networks_decentralized as pnd

# import supplement files
import plot_objectives as po
import plot_networks_all as pna
import plot_joint_grids_vs_baseline as pjg
import plot_random_forest_importances as prfi
import plot_correlation as pc
import plot_regression_coefficients as prc


if __name__ == '__main__':

    # Set dataset parameters
    execset = 10
    model = '3xx'
    base_graph = 'empty'
    steps = [0.001, 0.01, 0.1, 1]
    size = 9

    #%% Main text plots

    # Figure 3: Example tasks, networks, & difficulties
    ptnd.plot_tasks_networks_difficulties()
    
    # Reset fig settings
    fs.set_fonts()

    # Figure 4: Example run averages
    __ = per.plot_example_run_averages(execset)

    # Figure 5: Relative team performance plot
    pom.plot_outcome_means(execset, team_size=size, base_graph=base_graph)

    # Figures 6 & 7: Directional importances
    pit.plot_importances_tasks()
    pin.plot_importances_networks()
    
    # Figure 8: Decentralized networks
    pnd.plot_decentralized_graphs()


    #%% SI Appendix plots

    # Generate tasks
    po.plot_objective_pages()

    # Generate networks
    pna.plot_all_graphs()

    # Generate heatmaps
    for step in steps:
        pjg.plot_joint_grid(execset, model, base_graph, step)

    # Generate relative performances for all graphs
    pom.plot_page()
    
    # Plot regression coefficients
    prc.plot_regression_coefficients()
    prc.plot_regression_coefficients(reg='all')

    # Plot random forest importances
    prfi.plot_randfor_imps()
    
    # Plot correlations
    pc.plot_correlations()
