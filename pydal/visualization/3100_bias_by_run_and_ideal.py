# -*- coding: utf-8 -*-
"""

Visualize the results of scripts 2010/2013, MAE rMSE and bias calculation 

(bias is for an ideal run track)

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
RELATIVE = True

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
                        p_type = r'bias'):
    """
    p_type can be L2, L1, or bias (note lower cap)

    L2 (MSE) has its result square rooted.
    """
    cols = p_df.columns
    f = df[cols[0]].values + 10
    c = [ x for x in cols if p_hydro in x]
    c_select = [x for x in c if p_type in x]


    for c in c_select:
        lab = string_cleaner(c)
        if 'MSE' in c:
            p_ax.scatter(f,df[c].values**0.5,marker='.',label=lab)
        else:
            p_ax.scatter(f,df[c].values,marker='.',label=lab)
    p_ax.set_xlim([30,300])
    p_ax.set_xscale('log')
    if p_type =='MSE':
        p_ax.set_ylabel(r'$\sqrt{MSE}$',fontsize=_thesis.SIZE_AX_LABELS)
    else:
        p_ax.set_ylabel(p_type ,fontsize=_thesis.SIZE_AX_LABELS)
    p_ax.xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    p_ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    return p_ax


if __name__ == '__main__':
    
    fname           =  coordinate+'_dependent_L1_L2_bias_high_capacity.csv'
    fname_full      = directory + r'\\' + fname
    df = pd.read_csv(fname_full)
       
    if ABSOLUTE:
    
        # This is for legend use only - spot check once in a while too.
        cols = df.columns
        c = [ x for x in cols if hydro in x]
        c_select = [x for x in c if 'Bias' in x]
        
        fig,axs = plt.subplots(nrows=3,ncols=1,figsize = _thesis.FIGSIZE_TRIPLE_STACK )
        plot_result_by_type(axs[0],p_df = df,p_hydro = hydro,p_type='MAE')
        plot_result_by_type(axs[1],p_df = df,p_hydro = hydro,p_type='MSE')
        plot_result_by_type(axs[2],p_df = df,p_hydro = hydro,p_type='Bias')
        axs[0].set_ylim([2,7])
        axs[1].set_ylim([2,7])
        axs[2].set_ylim([-2,9])
        
        fig.supxlabel('Frequency (Hz)',fontsize=_thesis.SIZE_AX_LABELS)
        # fig.supylabel('Unit: dB ref $1{\mu}Pa^2m^2 / Hz$',fontsize = _thesis.SIZE_AX_LABELS)
        plt.legend(ncols=3,bbox_to_anchor=(0.5,3.65),loc='center')
        # plt.legend()
        
        figname =  'MAE_MSE_Bias_'+hydro+'_' +coordinate 
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

    if RELATIVE:
        # Same as absolute, except shows relative to the high capacity model
        
        
        cols = df.columns
        c = [ x for x in cols if hydro in x]
        c_MAE_ref = [x for x in c if '1_layers_38_nodes_MAE' in x][0]
        c_MSE_ref = [x for x in c if '1_layers_38_nodes_MSE' in x][0]
        MAE_ref = df[c_MAE_ref].values
        MSE_ref = df[c_MSE_ref].values
        f = df[cols[0]].values + 10
        
        fig,axs = plt.subplots(nrows=3,ncols=1,figsize = _thesis.FIGSIZE_TRIPLE_STACK )


        # What kind of metric:        
        metric_type = 'MAE'        
        c_select = [x for x in c if metric_type in x]
        c_legend = []
        for col in c_select:
            c_legend.append(string_cleaner(col))
        
        for col in c_select:
            if 'capacity' in c : continue
            lab = string_cleaner(col)
            if 'MSE' in col:
                delta = df[col].values**0.5 - MSE_ref**0.5
                axs[0].scatter(f,delta,marker='.',label=lab)
            else:
                delta = df[col].values - MAE_ref
                axs[0].scatter(f,delta,marker='.',label=lab)
        axs[0].set_xlim([30,300])
        axs[0].set_xscale('log')
        axs[0].set_ylabel(metric_type ,fontsize=_thesis.SIZE_AX_LABELS)
        axs[0].xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        axs[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())


        # What kind of metric:
        metric_type = 'MSE'

        c_select = [x for x in c if metric_type in x]
        c_legend = []
        for col in c_select:
            c_legend.append(string_cleaner(col))

        for col in c_select:
            if 'capacity' in col : continue
            lab = string_cleaner(col)
            if 'MSE' in col:
                delta = df[col].values**0.5 - MSE_ref**0.5
                axs[1].scatter(f,delta,marker='.',label=lab)
            else:
                delta = df[col].values - MAE_ref
                axs[1].scatter(f,delta,marker='.',label=lab)
        axs[1].set_xlim([30,300])
        axs[1].set_xscale('log')
        axs[1].set_ylabel(metric_type ,fontsize=_thesis.SIZE_AX_LABELS)
        axs[1].xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        axs[1].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())



        plot_result_by_type(axs[2],p_df = df,p_hydro = hydro,p_type='Bias')
        axs[0].set_ylim([-1.5,1.5])
        axs[1].set_ylim([-1.5,1.5])
        axs[2].set_ylim([-2,10])

        plt.legend(ncols=3,bbox_to_anchor=(0.5,3.65),loc='center')
        fig.supxlabel('Frequency (Hz)',fontsize=_thesis.SIZE_AX_LABELS)
        # fig.supylabel('Unit: dB ref $1{\mu}Pa^2m^2 / Hz$',fontsize = _thesis.SIZE_AX_LABELS)


        figname =  'MAE_MSE_Bias_RELATIVE_'+hydro+'_' +coordinate 
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
