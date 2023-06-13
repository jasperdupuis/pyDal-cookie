# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:18:22 2023

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


def compute_linear_regressions(
        p_data_dir,
        p_type,
        p_write_file = True):
    """
    linear regression and correlation analysis for each frequency bin and run.
    
    The dependent variable is received level
    
    The explanatory variables are: time (XY proxy), 20logR.
    
    There is also correlation of the two hydrophones with each other.
    
    result is keyed on runID.
    
    pass p_type = 'decibel' for 10log10(value), not usually wanted. Remember
    value in above is the value stored in hdf5 files, which are linear values.
    """
    list_runs = os.listdir(p_data_dir)
    list_runs = [ x.split('_')[0] for x in list_runs] 

    result = dict()
    for runID in list_runs:
        if 'summary' in runID: continue # not valid.
        if 'frequency' in runID: continue # not valid.
        if 'DR' not in runID: continue # only dynamic runs are valid. Shouldn't be possible but just in case.
        
        temp = dict()
        
        runData = \
            pydal.utils.load_target_spectrogram_data(runID,  # Returns linear values
                                                     p_data_dir)
        f = runData['Frequency'][_vars.INDEX_FREQ_LOW : _vars.INDEX_FREQ_HIGH ]
        r = runData['R']   # distance from CPA
        r = np.sqrt (r**2 + 100**2) # distance from hydrophone

        spec_n = \
            runData['South_Spectrogram'][_vars.INDEX_FREQ_LOW : _vars.INDEX_FREQ_HIGH , : ]
        spec_s = \
            runData['North_Spectrogram'][_vars.INDEX_FREQ_LOW : _vars.INDEX_FREQ_HIGH , : ]
        
        if p_type == 'decibel':
            spec_n = 10*np.log10(spec_n)
            spec_s = 10*np.log10(spec_s)
            
         # time, a proxy for x-y.
        corr_n = np.zeros_like(f)
        p_n = np.zeros_like(f)
        corr_s = np.zeros_like(f)
        p_s = np.zeros_like(f)        
        for index in range(len(f)):
            pearson = stats.pearsonr(
                runData['North_Spectrogram_Time'] ,
                runData['North_Spectrogram'][index] )            
            corr_n[index] = pearson.statistic
            p_n[index] =    pearson.pvalue
            pearson = stats.pearsonr(
                runData['South_Spectrogram_Time'] ,
                runData['South_Spectrogram'][index] )            
            corr_s[index] = pearson.statistic
            p_s[index] =    pearson.pvalue
            
        temp['North_v_time_r'] = corr_n
        temp['North_v_time_p'] = p_n
        temp['South_v_time_r'] = corr_s
        temp['South_v_time_p'] = p_s
    
    
        # 20logr
        # Check for r < and > of spec length
        # Linear values cast to dB for this transform only.
        dr = r[1] - r[0]
        while (len(r) < runData['North_Spectrogram'].shape[1]):
            r = list(r)
            r.append(r[-1] + dr)
        while (len(r) > runData['North_Spectrogram'].shape[1]):
            r = r[:-1]
        corr_n = np.zeros_like(f)
        p_n = np.zeros_like(f)
        corr_s = np.zeros_like(f)
        p_s = np.zeros_like(f)        
        for index in range(len(f)):
            pearson = stats.pearsonr(
                20*np.log10(r) ,
                10*np.log10(runData['North_Spectrogram'][index] ) )           
            corr_n[index] = pearson.statistic
            p_n[index] =    pearson.pvalue
            pearson = stats.pearsonr(
                20*np.log10(r) ,
                10*np.log10(runData['South_Spectrogram'][index] )  )      
            corr_s[index] = pearson.statistic
            p_s[index] =    pearson.pvalue

        temp['North_v_20log_r'] = corr_n
        temp['North_v_20log_p'] = p_n
        temp['South_v_20log_r'] = corr_s
        temp['South_v_20log_p'] = p_s

        # south hydro (X) corr with north hydro (Y)
        corr = np.zeros_like(f)
        p = np.zeros_like(f)        
        for index in range(len(f)):
            pearson = stats.pearsonr(
                runData['South_Spectrogram'][index] ,
                runData['North_Spectrogram'] [index])                
            corr[index] = pearson.statistic
            p[index] =    pearson.pvalue
        temp['North_v_South_r'] = corr
        temp['North_v_South_p'] = p
        result[runID] = temp
        
        print (runID + ' finished computing')

    result['Frequency'] = f
    
    if p_write_file:
        fname = p_data_dir + r'regress_and_corr_dict.pkl'
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
    
    regression_stats = compute_linear_regressions(
        p_data_dir = data_dir,
        p_type = 'linear',
        p_write_file = True)
