# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:45:58 2023

@author: Jasper
"""

import os
import pickle
import numpy as np
import scipy.stats as stats
import h5py as h5
import pandas as pd

import pydal
import pydal.utils
import pydal._variables as _vars
import pydal._directories_and_files as _dirs


def compute_summary_stats_hydrophones(
        p_data_dir,
        p_type,
        p_write_file = True):
    """
    Compute mean, std, SI, S, K for each frequency bin and run.
    
    result is keyed on runID.
    
    pass p_type = 'decibel' for 10log10(value), not usually wanted.
    """
    list_runs = os.listdir(p_data_dir)
    list_runs = [ x.split('_')[0] for x in list_runs] 

    result = dict()
    for runID in list_runs:
        if 'summary' in runID: continue # not valid.
        temp = dict()
        if 'frequency' in runID: continue # not valid.
        runData = \
            pydal.utils.load_target_spectrogram_data(runID,  # Returns linear values
                                                     p_data_dir)
        spec_n = \
            runData['South_Spectrogram'][_vars.INDEX_FREQ_LOW : _vars.INDEX_FREQ_HIGH , : ]
        spec_s = \
            runData['North_Spectrogram'][_vars.INDEX_FREQ_LOW : _vars.INDEX_FREQ_HIGH , : ]
        
        if p_type == 'decibel':
            spec_n = 10*np.log10(spec_n)
            spec_s = 10*np.log10(spec_s)
            
        # Compute mean SOG if dynamic run
        if 'AM' not in runID:
            x = runData['X'][:]
            y = runData['Y'][:]
            dx = np.zeros(len(x)-10)
            dy = np.zeros(len(x)-10)
            for index in range(len(dx)):
                dx[index] = x[index + 10] - x[index]
                dy[index] = y[index + 10] - y[index]
            r = np.sqrt( ( dx ** 2 ) + ( dy ** 2 ))
            sog = np.mean (r) #Because it's 10 samples, the number itself is SOG
            sog_std = np.std(r)
            temp['SOG_Mean_ms'] = sog     
            temp['SOG_STD_ms'] = sog_std
        
        # Compute moments of the hydrophone spectrograms
        m_n = np.mean(spec_n,axis=1)            
        std_n = np.std(spec_n,axis=1)
        s_n = stats.skew(spec_n,axis=1)
        k_n = stats.kurtosis(spec_n,axis=1)
        si_n = ( std_n ** 2 ) / (m_n ** 2)

        m_s = np.mean(spec_s,axis=1)
        std_s = np.std(spec_s,axis=1)
        s_s = stats.skew(spec_s,axis=1)
        k_s = stats.kurtosis(spec_s,axis=1)
        si_s = ( std_s ** 2 ) / (m_s ** 2)
        
        temp['Frequency'] = \
            runData['Frequency'][_vars.INDEX_FREQ_LOW : _vars.INDEX_FREQ_HIGH ]
        
        temp['North_Mean'] = m_n            
        temp['North_STD'] = std_n
        temp['North_Skew'] = s_n
        temp['North_Kurtosis'] = k_n
        temp['North_Scintillation_Index'] = si_n

        temp['South_Mean'] = m_s           
        temp['South_STD'] = std_s
        temp['South_Skew'] = s_s
        temp['South_Kurtosis'] = k_s
        temp['South_Scintillation_Index'] = si_s
        
        result[runID] = temp

    if p_write_file:
        fname = p_data_dir + r'summary_stats_dict.pkl'
        with open( fname, 'wb' ) as file:
            pickle.dump( result, file )

    return result


if __name__ == '__main__':    
    local_df = pd.read_csv(_dirs.TRIAL_MAP)
    list_run_IDs = local_df[ local_df.columns[1] ].values

    overlap = _vars.OVERLAP
    window_length = _vars.T_HYD_WINDOW
    window = np.hanning(_vars.FS_HYD * window_length)
    fs_hyd = _vars.FS_HYD

    local = pydal.utils.create_dirname(fs_hyd, len(window), overlap)
    data_dir = _dirs.DIR_SPECTROGRAM + local + '\\'
    
    summary_stats = compute_summary_stats_hydrophones(
        p_data_dir = data_dir,
        p_type = 'linear',
        p_write_file = True)

