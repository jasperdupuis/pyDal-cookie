# -*- coding: utf-8 -*-
"""

Variability analysis of the existing method ranges using RAM TL models

Must trim array based on available frequencies for which I ran RAM model

"""

import pandas as pd
import numpy as np
from scipy import interpolate

import pydal.utils
import pydal._variables as _vars
import pydal._directories_and_files as _dirs

from UWAEnvTools.singleTL.RAM import read_XY_TL_df


def process_run_RAM(
    p_gram_lin,
    p_f,
    p_gram_x,
    p_gram_y,
    p_dir_RAM   = _dirs.DIR_RAM_DATA,
    p_hydro     = _vars.HYDROPHONE,
    p_f_min_RAM = _vars.RAM_F_MIN_AVAIL,
    p_f_max_RAM = _vars.RAM_F_MAX_AVAIL ):
    """
    From already trimmed gram and xy data, calculate the avreage SL and STD.
    Do this accurately, using RAM model results:
        
        in dB, SL_i = RL_i + TL_i
        
        SL_lin_mean = mean (_vars.REF_UPA * 10** ( SL_i / 10 ) )
        
        SL_db_mean = 10*np.log10(SL_lin_mean / _vars.REF_UPA)
        
    
    For 10 to 599 Hz, takes just under 2 minutes.
    
    """
    
    RL          = 10 * np.log10( p_gram_lin/ _vars.REF_UPA)
    SL_dB       = RL
    
    # For available RAM freqs, interpolate TLs and apply to RL
    RAM_freqs = [ int(freq) for freq in p_f if (freq >= p_f_min_RAM) and ( freq < p_f_max_RAM ) ]
    failures = []
    for freq in RAM_freqs:
        # get the specific RAM TL model frequency index in the RL f basis
        RL_f_index      = pydal.utils.find_target_freq_index(freq, p_f )
        if freq in _vars.RAM_F_FAILS:
            # Leave RL unchanged but note the failure
            SL_dB [ RL_f_index , : ] =  RL[ RL_f_index , : ]  
            failures.append(freq)
            continue
        fname_TL        = p_dir_RAM + str(freq).zfill(4) + r'_' + p_hydro.capitalize() + '.csv'
        X,Y,TL          = read_XY_TL_df(fname_TL)
        
        TL_interped     = interpolate.griddata(
            ( X , Y ) ,
            TL ,
            ( p_gram_x , p_gram_y ) )
        SL_dB [ RL_f_index , : ] =  RL[ RL_f_index , : ]  + TL_interped
    
    # r_TL is an array of TLs for each XY step in the spectrogram.
    # array broadcasts so that each timestep's 20logR is applied to all freqs
    SL_lin      = _vars.REF_UPA * 10 ** ( SL_dB / 10 )
    mean_lin    = np.mean( SL_lin ,axis=1)
    mean_dB     = 10* np.log10 ( mean_lin / _vars.REF_UPA)
    std_dB      = np.std( SL_dB ,
                    axis = 1 )
    std_lin     = np.std( SL_lin ,
                    axis = 1 )
    
    return mean_dB, std_dB, std_lin


def setup_trim_and_process_run_RAM(
    p_runID         = r'DRJ3PB09AX01EB',
    p_distToCPA      = 33,
    p_dir_ram       = _dirs.DIR_RAM_DATA,
    p_dir_spec      = _dirs.DIR_SPECTROGRAM,
    p_fs_hyd        = _vars.FS_HYD,
    p_n_window      = _vars.T_HYD_WINDOW * _vars.FS_HYD,
    p_overlap       = _vars.OVERLAP
    ): # m,  defines the data window length (uses 2x this centered on CPA)
    
    dir_spec_subdir = pydal.utils.create_dirname_spec_xy(
        p_fs_hyd,
        p_n_window,
        p_overlap
        )
    dir_spec = p_dir_spec + dir_spec_subdir + '\\'

    # Get the gram and xy data, rotate the xy data.
    gram_dict,N       = pydal.utils.get_spectrogram_file_as_dict(p_runID, dir_spec)
    f               = gram_dict['Frequency']
    t               = gram_dict['North_Spectrogram_Time']
    spec_x,spec_y   = pydal.utils.rotate(gram_dict['X'], gram_dict['Y'], _vars.TRACK_ROTATION_RADS)
    if len( t ) < len(spec_x): # trim the last xy to make t_n fit.
        spec_x,spec_y = spec_x[:-1],spec_y[:-1] 
    gram_dict['X'] = spec_x
    gram_dict['Y'] = spec_y
    
    # Get the indices associated with the available RAM TL frequencies:
    start_f_index                   = pydal.utils.find_target_freq_index(_vars.RAM_F_MIN_AVAIL , f )
    end_f_index                     = pydal.utils.find_target_freq_index(_vars.RAM_F_MAX_AVAIL , f )
    gram_dict['North_Spectrogram']  = gram_dict['North_Spectrogram'][ start_f_index : end_f_index , : ]
    gram_dict['South_Spectrogram']  = gram_dict['South_Spectrogram'][ start_f_index : end_f_index , :]
    f                               = gram_dict['Frequency'][ start_f_index : end_f_index ]
    
    # Get the indices associated with the data window length:
    start_t_index , end_t_index     = pydal.utils.get_gram_XY_index_by_distToCPA(gram_dict,p_distToCPA)
    # trim the spectrograms and X Y data to only contain the data window length
    gram_dict['North_Spectrogram']  = gram_dict['North_Spectrogram'][ :, start_t_index : end_t_index ]
    gram_dict['South_Spectrogram']  = gram_dict['South_Spectrogram'][ :, start_t_index : end_t_index ]
    gram_dict['X']                  = gram_dict['X'] [start_t_index : end_t_index ]
    gram_dict['Y']                  = gram_dict['Y'] [start_t_index : end_t_index ]


    # North spectrogram processing RAM TL model
    n_SL_mean_dB, n_SL_std_dB, n_SL_std_lin = process_run_RAM(
        p_gram_lin  = gram_dict['North_Spectrogram'],
        p_f         = f,
        p_gram_x    = gram_dict['X'],
        p_gram_y    = gram_dict['Y'] 
        )
    
    # South spectrogram processing 20logR
    s_SL_mean_dB, s_SL_std_dB, s_SL_std_lin = process_run_RAM(
        p_gram_lin  = gram_dict['South_Spectrogram'],
        p_f         = f,
        p_gram_x    = gram_dict['X'],
        p_gram_y    = gram_dict['Y']
        )

    return f, n_SL_mean_dB, n_SL_std_dB, n_SL_std_lin ,s_SL_mean_dB, s_SL_std_dB, s_SL_std_lin 


def batch_process_and_store_runs_RAM(
        p_runlist,
        p_save_dir,
        p_dist_to_CPA   = _vars.DIST_TO_CPA,
        p_dir_ram       = _dirs.DIR_RAM_DATA,
        p_dir_spec      = _dirs.DIR_SPECTROGRAM,
        p_fs_hyd        = _vars.FS_HYD,
        p_n_window      = _vars.T_HYD_WINDOW * _vars.FS_HYD,
        p_overlap       = _vars.OVERLAP
        ):
    n_SL_mean_dB_dict = dict()
    n_SL_std_dB_dict = dict()
    n_SL_std_lin_dict = dict()
    s_SL_mean_dB_dict = dict()
    s_SL_std_dB_dict = dict()
    s_SL_std_lin_dict = dict()

    for runID in p_runlist:
        f, n_SL_mean_dB, n_SL_std_dB, n_SL_std_lin ,s_SL_mean_dB, s_SL_std_dB, s_SL_std_lin =\
            setup_trim_and_process_run_RAM(
                runID,
                p_dist_to_CPA,
                p_dir_ram ,
                p_dir_spec,
                p_fs_hyd  ,
                p_n_window,
                p_overlap )
        n_SL_mean_dB_dict[runID]    = n_SL_mean_dB
        n_SL_std_dB_dict[runID]     = n_SL_std_dB
        n_SL_std_lin_dict[runID]    = n_SL_std_lin
        s_SL_mean_dB_dict[runID]    = s_SL_mean_dB
        s_SL_std_dB_dict[runID]     = s_SL_std_dB
        s_SL_std_lin_dict[runID]    = s_SL_std_lin
    n_SL_mean_dB_dict['Frequency']  = f
    n_SL_std_dB_dict['Frequency']   = f
    n_SL_std_lin_dict['Frequency']  = f
    s_SL_mean_dB_dict['Frequency']  = f
    s_SL_std_dB_dict['Frequency']   = f
    s_SL_std_lin_dict['Frequency']  = f
    
    df = pd.DataFrame.from_dict(n_SL_mean_dB_dict)
    df.to_csv(p_save_dir + 'North Mean SL dB.csv')

    df = pd.DataFrame.from_dict(n_SL_std_dB_dict)
    df.to_csv(p_save_dir + 'North STD SL dB.csv')

    df = pd.DataFrame.from_dict(n_SL_std_lin_dict)
    df.to_csv(p_save_dir + 'North STD SL lin.csv')

    df = pd.DataFrame.from_dict(s_SL_mean_dB_dict)
    df.to_csv(p_save_dir + 'South Mean SL dB.csv')

    df = pd.DataFrame.from_dict(s_SL_std_dB_dict)
    df.to_csv(p_save_dir + 'South STD SL dB.csv')

    df = pd.DataFrame.from_dict(s_SL_std_lin_dict)
    df.to_csv(p_save_dir + 'South STD SL lin.csv')    
    
    return    

if __name__ == '__main__':
    
    import time
    import matplotlib.pyplot as plt


    # Process all runs that have passed spectrogram generation
    """
    dir_name_savefiles    = _dirs.DIR_SL_RESULTS_RAM
    dir_name_spectrograms = pydal.utils.create_dirname_spec_xy(
        p_fs              = _vars.FS_HYD,
        p_win_length      = _vars.T_HYD * _vars.FS_HYD,
        p_overlap_fraction= _vars.OVERLAP
        )
    dir_name_spectrograms = _dirs.DIR_SPECTROGRAM + dir_name_spectrograms
    run_list = pydal.utils.get_all_runs_in_dir(dir_name_spectrograms)
    batch_process_and_store_runs_RAM(
        run_list,
        dir_name_savefiles)
    """
    # Below runs and vsualized a single run.
    # Note code a bit different from 20logR to RAM, be careful.
#    """

    RUNID           = r'DRJ3PB09AX01EB'
    CPA_DIST        = 33
        
    start = time.time()

    f, n_SL_mean_dB, n_SL_std_dB, n_SL_std_lin ,s_SL_mean_dB, s_SL_std_dB, s_SL_std_lin  = \
        setup_trim_and_process_run_RAM(RUNID)

    end = time.time()
    print('time elapsed: \t' + str(end-start) + ' seconds')
    
    
    plt.figure()
    plt.title('Source level (SL) estimates with RAM \n Run ID: ' + RUNID)
    plt.plot(n_SL_mean_dB ,label='north') 
    plt.plot( s_SL_mean_dB ,label='south')
    plt.xscale('log')
    plt.legend()
    
    plt.figure()
    plt.title('Delta North - South with RAM \n SL Estimates\n Run ID: ' + RUNID)
    plt.xscale('log')
    plt.plot( n_SL_mean_dB - s_SL_mean_dB )
    
    plt.figure()
    plt.title('STD of SL dB array with RAM \n Run ID: ' + RUNID)
    plt.plot( f, n_SL_std_dB , label = 'North')
    plt.plot( f, s_SL_std_dB , label = 'South')
    plt.xscale('log')
    plt.legend()
    
    plt.figure()
    plt.title('10log ( STD of SL linear array ) with RAM \n Run ID: ' + RUNID)
    s_dB_of_SL_std_lin  = 10 * np.log10( s_SL_std_lin )
    n_dB_of_SL_std_lin  = 10 * np.log10( n_SL_std_lin )
    plt.plot( f,  n_dB_of_SL_std_lin , label = 'North')
    plt.plot( f,  s_dB_of_SL_std_lin , label = 'South')
    plt.xscale('log')
    plt.legend()
    
    plt.figure()
    plt.title('Delta North - South, with RAM \n 10log ( STD of SL linear array )\n Run ID: ' + RUNID)
    plt.plot( f,  n_dB_of_SL_std_lin - s_dB_of_SL_std_lin , label = 'North - South')
    plt.xscale('log')
    plt.legend()


#    """














