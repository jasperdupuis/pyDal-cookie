# -*- coding: utf-8 -*-
"""


Use a NN model to make an estimate of bias for real data set.
(Just need one NN, as they perform very similarly)

1) get a set of runs x-y-spectrogram (not all, but also maybe all?)

2) Prune to frequency range of interest

3) Get models over the frequency range

4) For each frequency, for each run :
        compute mean level twice:
            first with simple arithmetic average
            then with application of model correction
                (must first refer corr to the model value at ref (0,0))
"""


import torch
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import pydal
import pydal._directories_and_files as _dirs
import pydal._variables as _vars
import pydal._thesis_constants as _thesis

GENERATE_DATA   = False
VISUALIZE       = True

COORD               = 'Y'
N_LAYER             = 1 
N_NODE              = 14
LAYER               = str(N_LAYER) +' layers'
NODE                = str(N_NODE) + ' nodes'

Y_REF               = 0.0

RUN_SELECT_LIST     = [2,7,14,17,23,27,34]

FMIN                = 30
FMAX                = 301



def calculate_real_run_bias(p_data,p_hydro):
    f           = p_data['Frequency']
    rl_s        = p_data['South'] # 2d array, zero mean gram
    rl_n        = p_data['North'] # 2d array, zero mean gram
    # Add 100 to these to get back to non zmrl. 
    # Note this will make all speeds look the same!!!!
    # (On absolute scales)
    rl_s        = rl_s + 100
    rl_n        = rl_n + 100
    # Note this will make all speeds look the same!!!!
    
    x           = p_data['X'] / _vars.X_SCALING
    y           = p_data['Y'] / _vars.Y_SCALING
    runs        = p_data['Runs']
    run_lengths = p_data['Run_Lengths']
    
    f_min_ind   = pydal.utils.find_target_freq_index(FMIN, f)
    f_max_ind   = pydal.utils.find_target_freq_index(FMAX, f)
    
    frange      = f[f_min_ind:f_max_ind]
    rl_sf       = rl_s[f_min_ind:f_max_ind]
    rl_nf       = rl_n[f_min_ind:f_max_ind]
    
    
    """
    3) Load all models in to a dict
    """
    
    
    nn_dir      = _dirs.DIR_SINGLE_F_1D_NN + r'hdf5_spectrogram_bw_1.0_overlap_90' \
        + r'/' + p_hydro \
        + r'/' + COORD \
        + r'/' + LAYER \
        + r'/' + NODE + r'/'
    
    
    models_dict = dict()
    
    for freq in frange:
        ff          = int(freq)
        fname       = str(ff).zfill(4) + r'.trch'
        f_model     = nn_dir + fname
        models_dict[ff]  = pydal.models.classes.DeepNetwork_1d(N_LAYER,N_NODE)
        models_dict[ff].load_state_dict(torch.load(f_model))
        models_dict[ff].eval()
    
    L_S_NOW_LIST    = []
    L_S_COR_LIST    = []
    YYS_LIST        = []
    for i in range(len(run_lengths)-1): #-1 because of the nan value at end.
        start       = int(np.sum(run_lengths[:i]))
        end         = int(start + run_lengths[i])
        # plt.scatter(x[start:end],y[start:end])
        yy          = torch.tensor(y[start:end])
        YYS_LIST.append (np.array(yy))
        rl_nn       = rl_nf[:,start:end]    
        rl_ss       = rl_sf[:,start:end] # Not used.
        # Straight up arithmetic average as is done now.
        L_S_NOW     = np.mean(
            _vars.REF_UPA * 10 ** ( rl_nn / 10 ),# put in real domain
            axis=1 #along freq axis
            )
        L_S_NOW_LIST.append ( 10 * np.log10 ( L_S_NOW / _vars.REF_UPA ) )    
    
        # Now corrected by the learnt N_PL,var
        L_S_COR = np.zeros_like(frange)
        for ii in range(len(frange)):
            corr    = np.zeros_like(yy)
            for iii in range(len(yy)):
                z       = yy[iii]
                z       = z.float()
                z       = z.reshape((1,1))           
                corr[iii] = \
                    np.float64(
                        models_dict[frange[ii]].neural_net(z) * _vars.RL_SCALING
                        )
            # Need to refer to CPA (y = 0)
            ref_y   = torch.tensor(Y_REF).float().reshape((1,1))
            val_ref = models_dict[frange[ii]].neural_net(ref_y) * _vars.RL_SCALING
            corr    = corr - np.float64(val_ref)
            rr      = rl_nn[ii,:] - corr
            L_S_COR[ii]     = np.mean(
                _vars.REF_UPA * 10 ** ( rr / 10 ) # put in real domain
                )
        L_S_COR_LIST.append ( 10 * np.log10 ( L_S_COR / _vars.REF_UPA ) )    

    return L_S_NOW_LIST,L_S_COR_LIST


if GENERATE_DATA :
    """
    1) and 2) GET AND PRUNE X-Y-GRAM DATASET :
    """
    fname2019   = r'concatenated_data_2019.pkl'
    fname2020   = r'concatenated_data_2020.pkl'
    data2019    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2019)
    data2020    = pydal.models.SLR_with_transforms.load_concat_arrays(fname2020)
    dataALL     = pydal.utils.concat_dictionaries(data2019,data2020)
    
    data_list   = {'2019' : data2019, '2020' : data2020, 'All' : dataALL}
    del data2019,data2020,dataALL
    
    results = dict()
    results['Frequency (Hz)'] = np.arange(30,301)
    for hydro in ['NORTH','SOUTH']:
        for key,data in data_list.items():
            L_S_NOW_LIST,L_S_COR_LIST = \
                calculate_real_run_bias(
                    p_data  = data, 
                    p_hydro = hydro
                    )
    
            DELTA = []
            for n,c in zip(L_S_NOW_LIST,L_S_COR_LIST):
               DELTA.append( c - n )
            results[hydro + r'_' + key] = DELTA
            
    # df          = pd.DataFrame.from_dict(results,orient='index')
    fname       = COORD + '_dependent_real_run_bias.pkl'
    pydal.utils.dump_pickle_file(
        results, 
        p_data_dir = _dirs.DIR_RESULT , 
        p_fname = fname)


if VISUALIZE:
    
    
    # The real run bias data:
    fname           = COORD + '_dependent_real_run_bias.pkl'
    fname_full      = _dirs.DIR_RESULT + fname
    res_vis         = pydal.utils.load_pickle_file(
        p_data_dir  = _dirs.DIR_RESULT, 
        p_fname     = fname)
    f               = res_vis['Frequency (Hz)']
    keys            = list(res_vis.keys())[1:] # Drops the Frequency (Hz) key

    # The ideal run bias data:
    fname           =  COORD+'_dependent_L1_L2_bias_high_capacity.csv'
    fname_full      = _dirs.DIR_RESULT + r'\\' + fname
    df              = pd.read_csv(fname_full)
    cols            = df.columns
    cols            = [x for x in cols if 'Bias' in x]
    cs              = [x for x in cols if '14' in x]
    
    # make the north plot first:
    for local_hydro in ['North','South']:
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize = _thesis.FIGSIZE_LARGE )
    
        # The real data:
        for k,val in res_vis.items():
            if k == 'Frequency (Hz)' : continue
            k       = k.capitalize()
            k       = k.split('_')
            k       = k[0].capitalize() + r' ' + k[1] + ', real'
            if not( local_hydro in k ): continue
            arr     = np.array (val)        
            m       = np.mean(arr,axis=0)
            ax.scatter(f,m,marker='.',label=k)
        # Now the ideal data:
        ck      =  [x for x in cs if local_hydro in x][0]
        ax.scatter(f,df[ck][20:],marker = '1',label = local_hydro + ' all, ideal')
    
        ax.set_xlim([30,300])
        ax.set_xscale('log')
        ax.xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        fig.supxlabel('Frequency (Hz)',fontsize=_thesis.SIZE_AX_LABELS)
        fig.supylabel('Bias, db re $1 \mu$Pa / Hz',fontsize=_thesis.SIZE_AX_LABELS)
        plt.legend()
        plt.grid(which='both')
    
        figname =  local_hydro + '_' + COORD + '_bias_ideal_v_real' 
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
    

# DELTA_ARR = np.array(DELTA)
# d_mean = np.mean(DELTA_ARR,axis=0)
# plt.plot(frange,d_mean,label='DELTA');plt.xscale('log')




















