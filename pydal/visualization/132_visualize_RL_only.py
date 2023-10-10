# -*- coding: utf-8 -*-
"""

For a given frequency, visualize its received level as a function of 
time / R from north hydrophone. For now R is hard coded with +/- 141 on 
the entire time series, this should match the 200m track.


"""

import numpy as np
import time

import matplotlib.pyplot as plt

import pydal.utils
import pydal._variables as _vars
import pydal._directories_and_files as _dirs


def plot_and_show_RL_single_freq(
        p_freq,
        p_fs_hyd,
        p_n_window, 
        p_overlap,
        p_ave_len,
        p_type,
        p_mth,
        p_machine,
        p_speed,
        p_head):
    """
    Treatment of what to do with opposing headings is still a hardcoded option below
    
    For now, eastbound runs are just skipped entirely. 
    (they can reasonably be argued to be flipped or kept in as-is)
    """
    
    dir_spec_subdir = pydal.utils.create_dirname_spec_xy(
        p_fs_hyd,
        p_n_window,
        p_overlap
        )
    dir_spec = p_dir_spec + dir_spec_subdir + '\\'
    
    run_list = pydal.utils.get_all_runs_in_dir(dir_spec)
    run_list = pydal.utils.get_run_selection(
        run_list,
        p_type      = p_type,
        p_mth       = p_mth,
        p_machine   = p_machine,
        p_speed     = p_speed,
        p_head      = p_head)
    
    gram_dict,N       = pydal.utils.get_spectrogram_file_as_dict(run_list[0], dir_spec)
    f_basis           = gram_dict['Frequency']
    freq_index        = pydal.utils.find_target_freq_index(p_freq, f_basis)
    
    
    # Result concatenation
    # hydro   = p_hydro.capitalize()
    hydro   = p_hydro.capitalize()
    x       = []
    y       = []
    t       = []
    RL      = []
    runs    = []
    for runID in run_list:
        gram_dict,N       = pydal.utils.get_spectrogram_file_as_dict(runID, dir_spec)
        t.append(gram_dict[ hydro + '_Spectrogram_Time'])
        rx = 10*np.log10(gram_dict[ hydro +'_Spectrogram'][ freq_index , : ]/_vars.REF_UPA)
        # RL.append(rx - np.mean(rx))
        RL.append(rx)
        runs.append(runID)
        
    # N_AVE=71
    ave_kernel = np.ones( p_ave_len ) / p_ave_len
    fig, axs = plt.subplots(1)
    fig.suptitle('Westbound RL for ' + str(p_freq) + ' Hz\n' + hydro + ' hydrophone')
    for run,time,rx in zip(runs,t,RL):
        rx = np.convolve( rx , ave_kernel , mode='same')
        Y = time - np.mean(time)
        Y = Y / np.max(Y) 
        Y = Y * 100
        if run[-2] == 'E': # Special treatment for eastbound runs - keep, flip, or ignore them.
            # rx = rx               # keep them unmodified
            continue                # skip them
            # rx = rx [::-1]        # flip them
            
        axs.plot( Y , rx , label = run )
    
    fig.supxlabel('Distance from CPA (m)')
    axs.legend()
    
    return fig,axs,run_list


if __name__ == '__main__':   

    N_AVE           = 11 #for a smoothing window
    p_freq          = 53
    # p_hydro         = _vars.HYDROPHONE
    p_hydro         = 'NORTH'
    p_dist_to_CPA   = _vars.DIST_TO_CPA
    p_location      = _vars.LOCATION
    p_dir_spec      = _dirs.DIR_SPECTROGRAM
    p_fs_hyd        = _vars.FS_HYD
    p_n_window      = _vars.T_HYD_WINDOW * _vars.FS_HYD
    p_overlap       = _vars.OVERLAP

    start = time.time()
    fig,axs,run_list = plot_and_show_RL_single_freq(
        p_freq,
        p_fs_hyd    = _vars.FS_HYD,
        p_n_window  = p_n_window, 
        p_overlap   = p_overlap,
        p_ave_len   = N_AVE,
        p_type      = 'DR',
        p_mth       = 'J',
        p_machine   = 'X',
        p_speed     = '09',
        p_head      = 'X')

    plt.show()

    end = time.time()
    print('single freq RL visualization : ' + str(end-start))