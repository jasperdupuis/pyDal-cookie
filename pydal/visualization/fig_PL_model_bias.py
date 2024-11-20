# -*- coding: utf-8 -*-
"""

visualize the pat bay bias results for north and south separately from the other
bias / l1 / l2 error results graph

"""


import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

#import models directory
import sys
sys.path.insert(1, r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal-cookie\pydal')
import models

import pydal._directories_and_files as _dirs
import pydal._variables as _vars
import pydal._thesis_constants as _thesis


#
#
#

ABSOLUTE = True


coordinate      = 'Y'
hydro           = 'North'
directory       = _dirs.DIR_RESULT

#
#
#

def string_cleaner(p_str):
    """
    Classes of strings: SLR, NN
    Each class has diferences: 2019, 2020, all vs N Nodes
    Each class has fixed possibilities: L1, L2, bias
    """
    if 'PL' in p_str:
        return p_str.split('_')[0]
    if 'SLR' in p_str:
        strs    = p_str.split('_')
        result  = strs[0] + ' ' \
                + strs[1].capitalize() 
        return result
    if 'NN' in p_str:
        if 'high_capacity' in p_str: 
            return 'NN High Capacity'
        strs    = p_str.split('_')
        result  = strs[0] + ' ' \
                + strs[4].capitalize() + ' ' \
                + strs[5].capitalize() + ' ' 
        return result
    return 'string_cleaner in 3100 did not work'

def plot_result_by_type(p_ax,
                        p_df,
                        p_hydro,
                        p_type = r'bias',
                        p_marker='.'):
    """
    p_type can be L2, L1, or bias (note lower cap)

    L2 (MSE) has its result square rooted.
    """
    cols = p_df.columns
    f = p_df[cols[0]].values + 10
    c = [ x for x in cols if p_hydro in x]
    c_select = [x for x in c if p_type in x]

    for c in c_select:
        lab = string_cleaner(c)
        if lab == 'BELL' : lab = 'BELLHOP'
        if lab == 'KRAK' : lab = 'KRAKEN'
        p_ax.scatter(f,p_df[c].values,marker=p_marker,label=lab)
    p_ax.set_xlim([30,300])
    p_ax.set_xscale('log')
    p_ax.xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    p_ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    return p_ax


if __name__ == '__main__':
    
    fname_pl_mods   = r'Y_'+hydro+'_dependent_bias_propagation_models.csv'
    fname_pl_full   = directory + r'\\' + fname_pl_mods
    
    df_pl   = pd.read_csv(fname_pl_full)
       
    if ABSOLUTE:
    
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize = _thesis.FIGSIZE_TRIPLE_STACK )

        plot_result_by_type(ax,p_df = df_pl,p_hydro = hydro,p_type='Bias',p_marker='.')
        ax.set_ylim([-20,20])
        
        fig.supxlabel('Frequency (Hz)',fontsize=_thesis.SIZE_AX_LABELS)
        fig.supylabel('Bias, dB ref $1{\mu}Pa^2 / Hz$',fontsize = _thesis.SIZE_AX_LABELS)

        # plt.legend(ncols=3,bbox_to_anchor=(0.5,3.65),loc='center')
        plt.legend()
        
        figname =  'PL_model_bias_'+hydro 
        plt.savefig(fname = _dirs.DIR_RESULT_RESIDUALS_AND_BIAS \
                    + figname +'.eps',
                    bbox_inches='tight',
                    format='eps',
                    dpi = _thesis.DPI)    
        plt.savefig(fname = _dirs.DIR_RESULT_RESIDUALS_AND_BIAS \
                    + figname +'.pdf',
                    bbox_inches='tight',
                    format='pdf',
                    dpi = _thesis.DPI)
        plt.savefig(fname = _dirs.DIR_RESULT_RESIDUALS_AND_BIAS \
                    + figname +'.png',
                    bbox_inches='tight',
                    format='png',
                    dpi = _thesis.DPI)
        plt.close('all')
        
        """
        # Plot a delta for MSE between high capacity and 38 node model.
        # Hard coded numbers etc for now.
        f               = df[cols[0]].values + 10
        c_mse_slr       = cols[85]
        c_mse_nn        = cols[-1]
        slr,nn          = df[c_mse_slr],df[c_mse_nn]
        delta       = (df[c_mse_nn].values**0.5) - (df[c_mse_slr].values**0.5)
        plt.figure();plt.scatter(f,delta,marker='.',label='rmse delta: nn - slr')
        plt.xscale('log')
        plt.legend()
        """

