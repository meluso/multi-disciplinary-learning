# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 14:41:53 2021

@author: John Meluso
"""
import datetime as dt
import itertools as it
import matplotlib.pyplot as plt
import model_team as mt
import pandas as pd
import seaborn as sns
import statsmodels.stats.api as sms
from matplotlib.backends.backend_pdf import PdfPages    


if __name__ == '__main__':
    
    # Team test parameters
    test = 5
    graphs = {'empty','complete','small_world','random','power'}
    sizes = {10, 25, 100}
    fncl2fnsb = {
        'root': {'nodewt','degreewt'},
        'abs_sum': {'nodewt','degreewt'},
        'sphere': {'nodewt','degreewt'},
        'sin2': {'nodewt','degreewt'},
        'sin2sphere': {'nodewt','degreewt'},
        'test': {'posexpdeg','negexpdeg'}
        }
    runs = 5
    steps = 5
    props = {'unwt','wt'}
    
    # Short name of graph2size2fncl2fnsb2step2prop
    cats2data = {}
    cats2results = {}
    for gf in graphs:
        cats2data[gf] = {}
        cats2results[gf] = {}
        for nd in sizes:
            cats2data[gf][nd] = {}
            cats2results[gf][nd] = {}
            for fncl in fncl2fnsb:
                cats2data[gf][nd][fncl] = {}
                cats2results[gf][nd][fncl] = {}
                for fnsb in fncl2fnsb[fncl]:
                    cats2data[gf][nd][fncl][fnsb] = {}
                    cats2results[gf][nd][fncl][fnsb] = {}
                    for st in range(steps + 1):
                        cats2data[gf][nd][fncl][fnsb][st] = {}
                        cats2results[gf][nd][fncl][fnsb][st] = {}
                        for pr in props:
                            cats2data[gf][nd][fncl][fnsb][st][pr] = []
    
    # Create a team
    for gf, nd, fncl, run \
        in list(it.product(graphs, sizes, fncl2fnsb, range(runs))):
        for fnsb in fncl2fnsb[fncl]:
            team = mt.Team(gf,nd,fncl,fnsb)
            for pr in props:
                for st in range(steps + 1):
                    if st > 0: team.step()
                    prop = team.get_fxs_avg(pr)
                    cats2data[gf][nd][fncl][fnsb][st][pr].append(prop)
                    
    
    #%% Calculate mean differencces
    
    for gf, nd, fncl, st, pr \
        in list(it.product(graphs,sizes,fncl2fnsb,range(steps+1),props)):
        for fnsb in fncl2fnsb[fncl]:
            if gf != 'empty':
                base = sms.DescrStatsW(
                    cats2data['empty'][nd][fncl][fnsb][st][pr])
                cmpr = sms.DescrStatsW(
                    cats2data[gf][nd][fncl][fnsb][st][pr])
                cats2results[gf][nd][fncl][fnsb][st]['mean_diff'] \
                    = cmpr.mean - base.mean
                ci = sms.CompareMeans(base,cmpr).tconfint_diff()
                cats2results[gf][nd][fncl][fnsb][st]['ci_lo'] = ci[0]
                cats2results[gf][nd][fncl][fnsb][st]['ci_hi'] = ci[1]
            
    
    
    #%% Construct baseline from empty graph
    pdf = pdf = PdfPages('../figures/run_test_' + str(test) + '.pdf')
    
    fig = plt.figure()
    axs = fig.subplots(len(sizes),len(fncl2fnsb))
    
    for row, nodes in enumerate(sizes):
        for col, fncol in enumerate(fncl2fnsb):
            for step in range(steps + 1):
                data_base = df.loc[(df.fn_class == fncl) & (df.nodes == nodes)
                                   & (df.step == step)]
            data = df.loc[(df.fn_class == fncl) & (df.nodes == nodes)]
            
    pdf.close()
    