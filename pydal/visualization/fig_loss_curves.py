# -*- coding: utf-8 -*-
"""

From pydal._variables, recall the split is 0.8 / 0.1 / 0.1

"""

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
from matplotlib import cm

import pydal
import pydal._variables as _vars
import pydal._directories_and_files as _dirs
import pydal._thesis_constants as _thesis


GENERATE_DATA   = True
LOAD_DATA       = True
VISUALIZE       = True

FREQS           = np.arange(30,301)
NODES           = [14]#,20,26,32,38]
N_PERCENTILES   = 10

hydro = 'SOUTH'
coord = 'Y'
layers = '1 layers'

def percentiles_from_array(p_array,
                           p_n_percentiles  = 10,
                           p_min_percentile = 0,
                           p_max_percentile = 90):
    """
    This operates along the first axis of p_array.
    
    For the loss data, need to take the transpose of p_array before passing.
    
    The percentiles are capped at 90th, dont care about outlying values.
    """
    
    percentiles = np.linspace(p_min_percentile,
                              p_max_percentile,
                              p_n_percentiles)

    len_axis_0   = p_array.shape[0]
    SDist  = np.zeros((len_axis_0 ,p_n_percentiles))
    for i in range(p_n_percentiles):
        for t in range(len_axis_0 ):
          SDist[ t , i ] = np.percentile(p_array[t,:],percentiles[i])

    return SDist


def plot_percentiles_from_pct_array(p_percentiles_t,
                                    p_percentiles_v,
                                    p_cm = cm.Reds,
                                    p_cm_v = cm.Blues
                                    ):

    fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=_thesis.FIGSIZE)

    # Test values first:
    len_axis = p_percentiles_t.shape[0]
    N = p_percentiles_t.shape[1]
    half = int((N-1)/2)
    ax1.plot(np.arange(0,len_axis,1), p_percentiles_t[:,half],color='k')
    for i in range(half):
        ax1.fill_between(np.arange(0,len_axis,1), p_percentiles_t[:,i],p_percentiles_t[:,-(i+1)],color=p_cm(i/half))

    # Validation values second:
    len_axis = p_percentiles_v.shape[0]
    N = p_percentiles_v.shape[1]
    half = int((N-1)/2)
    ax1.plot(np.arange(0,len_axis,1), p_percentiles_v[:,half],color='k',linestyle='dashdot')
    # for i in range(half):
    #     ax1.fill_between(np.arange(0,len_axis,1), p_percentiles_v[:,i],p_percentiles_v[:,-(i+1)],color=p_cm_v(i/half))
    

    ax1.tick_params(labelsize=_thesis.FONTSIZE)
    ax1.set_xlabel('Training batch number', fontsize=_thesis.FONTSIZE)
    ax1.set_ylabel('Loss', fontsize=_thesis.FONTSIZE)
    fig.tight_layout()

    return fig,ax1


def build_results_dict(
        p_hydro = hydro,
        p_nodes = NODES,
        p_freqs = FREQS,
        p_coord = coord,
        p_layers = layers,
        p_dir_origin = _dirs.DIR_SINGLE_F_1D_NN_MULTI_EPOCH):
    """
    
    """
    results_v = dict()
    results_t = dict()
    for node in p_nodes:
        nodestr = str(node) + ' nodes'
        # results will go here before dictionary assignment
        loss_v = []
        loss_t = []
        for freq in FREQS:
            freqstr = str(freq).zfill(4)
            
            single_f_nn_root = p_dir_origin+ r'hdf5_spectrogram_bw_1.0_overlap_90/'
            target_dir = single_f_nn_root \
                + hydro +r'/' \
                + coord + r'/' \
                + layers + r'/' \
                + nodestr +r'/' \
                + 'losses/'
                
            fname = freqstr +r'.loss'
            losses = pydal.utils.load_pickle_file(
                p_data_dir = target_dir, 
                p_fname = fname)
            
            local_tt = []
            local_vv = []
            for epoch in range(40):
                tt = losses['Train'][epoch] # all epoch batch training results
                local_tt.append(tt)
                
                vv = np.mean(losses['Test'][epoch]) # epoch result
                local_vv.append(vv)
            
            # Put each batch loss in to a single vector for later analysis
            local_ttt = []
            for sublist in local_tt:
                for item in sublist:
                    local_ttt.append(float(item.cpu().detach().numpy()))
                        
            loss_v.append ( local_vv ) # the epoch validation losses, averaged over the flattened axis
    
            loss_t.append ( local_ttt ) # the training batch losses flattened
    
        loss_t_arr = np.array(loss_t )
        loss_v_arr = np.array(loss_v)
    
        results_v[node] = loss_v_arr
        results_t[node] = loss_t_arr

    return results_t,results_v


if __name__ == '__main__':


    if GENERATE_DATA:
        for hydro in ['NORTH','SOUTH']:
            data_t,data_v= build_results_dict(hydro)
            pydal.utils.dump_pickle_file(data_t, 
                                         _dirs.DIR_RESULT_LOSS_CURVES, 
                                         hydro + r'_' + 'test_loss_multiepoch.pkl')
            pydal.utils.dump_pickle_file(data_v, 
                                         _dirs.DIR_RESULT_LOSS_CURVES, 
                                         hydro + r'_' + 'val_loss_multiepoch.pkl')


    if LOAD_DATA :
        fname_t = hydro + r'_' + 'test_loss_multiepoch.pkl'
        fname_v = hydro + r'_' + 'val_loss_multiepoch.pkl'
        
        data_t = pydal.utils.load_pickle_file(
            _dirs.DIR_RESULT_LOSS_CURVES, 
            fname_t)
        data_v = pydal.utils.load_pickle_file(
            _dirs.DIR_RESULT_LOSS_CURVES, 
            fname_v)
        
    if VISUALIZE:
        node = 14
        # node = 14
        working_arr_t = data_t[node]
        working_arr_v = data_v[node]
        
        # need to transpose the test results before passing to for reasons
        percentiles_v   = percentiles_from_array(working_arr_v.T,N_PERCENTILES)
        ## Fornorth hydrophone with data 0.8 :
        # percentiles_t   = percentiles_from_array(working_arr_t.T[::618],N_PERCENTILES)
        ## For south hydrophone with data 0.05:
        percentiles_t   = percentiles_from_array(working_arr_t.T[::39],N_PERCENTILES)
        fig_t,ax_t      = plot_percentiles_from_pct_array(percentiles_t,percentiles_v)

        ax_t.set_ylim([0,0.04])
        
        target_dir = _dirs.DIR_RESULT_LOSS_CURVES
        figname = hydro.capitalize() + r'_' + str(node) +'_layer_loss_curve_multiepoch'
        plt.savefig(fname = _dirs.DIR_RESULT_LOSS_CURVES \
                    + figname +'.pdf',
                    bbox_inches='tight',
                    format='pdf',
                    dpi = _thesis.DPI)
        plt.savefig(fname = _dirs.DIR_RESULT_LOSS_CURVES \
                    + figname +'.png',
                    bbox_inches='tight',
                    format='png',
                    dpi = _thesis.DPI)
        plt.close('all')

